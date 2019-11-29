#!/usr/bin/env python2.7
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function, division

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
# from time 
import time

import cv2
from openvino.inference_engine import IENetwork, IECore

import numpy as np

import rospy
from std_msgs.msg import String,UInt8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from yolo_ros_real_pose.msg import RealPose
from yolo_ros_real_pose.msg import ObjectsRealPose

''' Parameters for camera '''
fx = 611.855712890625
fy = 611.8430786132812
cx = 317.46136474609375
cy = 247.88717651367188

''' Global variables '''
rgb_image=[]
depth_img=[]
rgb_img_update=0
depth_img_update=0
sig_info_update=0
args = []
model_xml=[]
model_bin=[]
ie=[]
net=[]
input_blob=[]
exec_net=[]
labels_map=[]
time_stamp=[]

bridge = CvBridge()

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

xml_path =  sys.path[0].rsplit("/",1)[0] + "/model/frozen_darknet_yolov3_model.xml"
labels_path =  sys.path[0].rsplit("/",1)[0] + "/model/coco.names"

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      default=xml_path, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with "
                                            "camera)", required=False, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is MYRIAD", default="MYRIAD", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=labels_path, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.3, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.2, type=float)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False,
                      action="store_true")
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    return parser


class YoloV3Params:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])   
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            #print(mask)
            #mask=[[3, 4, 5], [0, 1, 2]]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            x = (col + predictions[box_index + 0 * side_square]) / params.side * resized_image_w
            y = (row + predictions[box_index + 1 * side_square]) / params.side * resized_image_h
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            w = w_exp * params.anchors[2 * n]
            h = h_exp * params.anchors[2 * n + 1]
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h / resized_image_h, w_scale=orig_im_w / resized_image_w))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union



def detect_and_give_real_pose(current_total_dect, current_image,current_depth_img):
    global fx,fy,cx,cy,stage,time_stamp, bridge
    
    # cv2.imshow("Image window", current_image)
    # cv2.waitKey(3)
    
    '''Copy current image''' 
    rgb_img_get = current_image.copy()
    depth_img_get = current_depth_img.copy()

    img_to_pub = current_image.copy()
    current_img_to_pub = bridge.cv2_to_imgmsg(img_to_pub, encoding="passthrough")

    ''' Hand made time synchronizer'''
    current_img_to_pub.header = time_stamp 
    current_total_dect.header=time_stamp

    ''' Prepare to input '''
    (rows,cols,channels) = current_image.shape
    n, c, h, w = net.inputs[input_blob].shape

    is_async_mode = False
    wait_key_code = 0
 
    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    request_id = cur_request_id
    in_frame = cv2.resize(rgb_img_get, (w, h))

    # resize input_frame to network size
    in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((n, c, h, w))

    try:
        exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})  ###### Most time is consumed here
    except:
        rospy.loginfo("Start async failed!")
        cv2.imshow("DetectionResults", rgb_img_get)
        cv2.waitKey(1)
        return current_img_to_pub, False

    # Collecting object detection results
    objects = list()
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        try:
            output = exec_net.requests[cur_request_id].outputs
        except:
            rospy.loginfo("Process failed!")
            cv2.imshow("DetectionResults", rgb_img_get)
            cv2.waitKey(1)
            return current_img_to_pub, False

        for layer_name, out_blob in output.items():
            layer_params = YoloV3Params(net.layers[layer_name].params, out_blob.shape[2])
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                         rgb_img_get.shape[:-1], layer_params,
                                         args.prob_threshold) 

    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                objects[j]['confidence'] = 0

    # Drawing objects with respect to the --prob_threshold CLI parameter
    objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

    if len(objects):   # Check number of objects
        if args.raw_output_message:
            log.info("\nDetected boxes for batch {}:".format(1))
            log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")
    else:
        # rospy.loginfo("Found nothing!")
        cv2.imshow("DetectionResults", rgb_img_get)
        cv2.waitKey(1)
        return current_img_to_pub, False

    origin_im_size = rgb_img_get.shape[:-1]
    #print(objects)


    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (int(min(obj['class_id'] * 15, 255)),
                 min(obj['class_id'] * 20, 255), min(obj['class_id'] * 25+100, 255))
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        if args.raw_output_message:
            log.info(
                "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                          obj['ymin'], obj['xmax'], obj['ymax'],
                                                                          color))

        cv2.rectangle(rgb_img_get, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
        #rospy.loginfo("detect "+det_label)
        cv2.putText(rgb_img_get,
                    "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        x_pos=0
        y_pos=0
        z_pos=0

        info=RealPose()
        info.label = det_label
        info.confidence = obj['confidence']
        info.pix_lt_x = obj['xmin']
        info.pix_lt_y = obj['ymin']
        info.pix_rb_x = obj['xmax']
        info.pix_rb_y = obj['ymax']

        ### Now calculate the real position
        if len(depth_img_get)!=0 and (obj['xmax']-obj['xmin'])*(obj['ymax']-obj['ymin'])>=0:
            ### Calculate position here by depth image and camera parameters
            depth_box_width=obj['xmax']-obj['xmin']
            depth_box_height=obj['ymax']-obj['ymin']
            delta_rate=0.3
            x_box_min=int(obj['xmin'] + depth_box_width*delta_rate)
            y_box_min=int(obj['ymin'] + depth_box_height*delta_rate)
            x_box_max=int(obj['xmax'] - depth_box_width*delta_rate)
            y_box_max=int(obj['ymax'] - depth_box_height*delta_rate)
            after_width=(depth_box_width*(1-2*delta_rate))
            after_height=(depth_box_height*(1-2*delta_rate))
            '''  '''
            rect = depth_img_get[y_box_min:y_box_max,x_box_min:x_box_max] * 0.001
            rect[np.where(rect == 0)] = 99
            print(rect.min())
            # z_pos=bb.sum()/(after_width*after_height)*0.001

            # rect = depth_img_get[obj['xmin']:obj['xmax'], obj['ymin']:obj['ymax']]
            # rect = rect

            ''' Using EM, too slow '''
            # print("In Em")
            # em_start_time = time.time()
            # EM_model = cv2.ml.EM_create()
            # EM_model.setClustersNumber(2)
            # EM_model.setCovarianceMatrixType(0)
            # EM_model.setTermCriteria((cv2.TermCriteria_MAX_ITER, 1, 0.2))  # int type, int maxCount, double epsilon
            # EM_model.trainEM(rect)
            # print(time.time() - em_start_time)
            # print("EM_model works fine")

            x_pos = (0.5 * (obj['xmax'] + obj['xmin']) - cx) * z_pos / fx
            y_pos = (0.5 * (obj['ymax'] + obj['ymin']) - cy) * z_pos / fy

            info.x=x_pos
            info.y=y_pos
            info.z=z_pos

        current_total_dect.result.append(info)

    cv2.imshow("DetectionResults", rgb_img_get)
    cv2.waitKey(1)

    return current_img_to_pub, True



def load_model():
    global args,model_xml,model_bin,ie,net,input_blob,exec_net,labels_map
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    rospy.loginfo("Creating Inference Engine...")
    ie = IECore()

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    rospy.loginfo("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    rospy.loginfo("Preparing inputs")
    input_blob = next(iter(net.inputs))

    #  Defaulf batch_size is 1
    net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    rospy.loginfo("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    rospy.loginfo("Loaded model")


def detection_loop():
        rate = rospy.Rate(100)
        global rgb_image, rgb_img_update,depth_img,depth_img_update, bridge

        pub_total=rospy.Publisher("/yolo_ros_real_pose/detected_objects",ObjectsRealPose,queue_size = 1)
        corresponding_img_pub=rospy.Publisher("/yolo_ros_real_pose/img_for_detected_objects",Image,queue_size = 1)

        while not rospy.is_shutdown():
            
            if rgb_img_update == 1 and depth_img_update==1:
                if len(rgb_image)!=0 and len(depth_img)!=0:
                    total_dect = ObjectsRealPose()

                    try:
                        detection_start_time = time.time()
                        img_to_pub, if_detected = detect_and_give_real_pose(total_dect, rgb_image, depth_img)  #detection function
                        rospy.loginfo("detection time =" + str(time.time()-detection_start_time))

                        if if_detected:
                            pub_total.publish(total_dect)
                            corresponding_img_pub.publish(img_to_pub)
                    except:
                        rospy.loginfo("detection error!")
                        continue

                    rgb_img_update = 0
                    depth_img_update =0
           
            rate.sleep()


class SubscriberClass:
    def __init__(self):
        self.depth_img=None
        self.color_img=None

    def color_callback(self,msg):
        global rgb_image,rgb_img_update, bridge
        self.color_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_image=self.color_img.copy()
        rgb_img_update=1

    def depth_callback(self,msg):
        global depth_img,depth_img_update,time_stamp, bridge
        self.depth_img=bridge.imgmsg_to_cv2(msg, "32FC1")
        depth_img=self.depth_img.copy()
        time_stamp=msg.header
        depth_img_update=1
        # rospy.loginfo("depth received!!")


def main():
    rospy.init_node('detection', anonymous=False,log_level=rospy.INFO)
    sub_callbacks=SubscriberClass()

    image_sub = rospy.Subscriber("/camera/color/image_raw",Image,sub_callbacks.color_callback)
    image_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",Image,sub_callbacks.depth_callback)

    try:
        load_model()
    except :
        # rospy.loginfo("load model failed,please restart !!!")
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            rospy.loginfo("load failed!!!")
            rate.sleep()
        return
    detection_loop()


if __name__ == '__main__':
    main()

