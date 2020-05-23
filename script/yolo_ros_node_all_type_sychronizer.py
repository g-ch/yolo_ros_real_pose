#!/usr/bin/env python
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
from time import time

import cv2
from openvino.inference_engine import IENetwork, IECore

import numpy as np
from Queue import Queue

import rospy
from std_msgs.msg import String,UInt8, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, PoseStamped, Pose
from control_msgs.msg import JointControllerState

from yolo_ros_real_pose.msg import RealPose
from yolo_ros_real_pose.msg import ObjectsRealPose

''' Parameters for camera '''
''' 640x480 '''
#fx = 611.855712890625
#fy = 611.8430786132812
#cx = 317.46136474609375
#cy = 247.88717651367188

''' 424x240 '''
fx = 307.04815673828125
fy = 307.09002685546875
cx = 215.5561981201172
cy = 119.21048736572266

''' Simulation '''
# fx = 205.46963709898583
# fy = 205.46963709898583
# cx = 320.5
# cy = 240.5

''' Global variables '''
rgb_image=[]
depth_img=[]
rgb_img_update=0
depth_img_update=0
sig_info_update=0
time_stamp_header=[]

motor_yaw_corrected = 0
motor_yaw_init = 0
if_init_motor = True

uav_pose = PoseStamped()
bridge = CvBridge()


''' Parameters to set '''
xml_path =  sys.path[0].rsplit("/",1)[0] + "/model/frozen_darknet_yolov3_model.xml"
labels_path =  sys.path[0].rsplit("/",1)[0] + "/model/voc.names"

device_set = "HDDL"  #CPU, GPU, FPGA, HDDL or MYRIAD. Note GPU only supports intel GPU
input_set = "ROS"

#real world
rgb_image_topic = "/camera/color/image_raw"
depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
motor_pose_topic = "/place_velocity_info"
position_topic = "/mavros/local_position/pose"

# Simulation
# rgb_image_topic = "/iris/vi_sensor/camera_depth/camera/image_raw"  #"/camera/color/image_raw"
# depth_image_topic = "/iris/vi_sensor/camera_depth/depth/disparity" #"/camera/depth/image_rect_raw"
# motor_pose_topic = "/iris/joint1_position_controller/state"  #"/place_velocity_info"
# position_topic = "/iris/ground_truth/pose"  #"/mavros/local_position/pose"

motor_yaw_syncronize = True # if set true, you may need a fake publisher when running test: "rostopic pub -r 50 /place_velocity_info geometry_msgs/Point32 0.3 0.3 0.3"
yaw_queue_size = 3 # about: camera_delay / 15ms, 15ms is the estimated delay for motor position estimation
motor_yaw_queue = Queue(yaw_queue_size) # This queue is to syncronize the time between camera and the motor

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Path to an .xml file with a trained model.",
                      default=xml_path,type=str)
    args.add_argument("-i", "--input", help="Required. Path to a image/video file. (Specify 'cam' to work with "
                                            "camera)",  type=str, default=input_set, required=False) #required=True, 
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU",  type=str, default=device_set, required=False)
    args.add_argument("--labels", help="Optional. Labels mapping file", type=str, default=labels_path)
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


class YoloParams:
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
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


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
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
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


class InfoGather:
    def __init__(self):
        self.uav_pose = Pose()
        self.motor_yaw = 0
        self.img_to_pub = []

def main():
    global args
    args = build_argparser().parse_args()
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    print("model loaded")

    # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------
    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
    log.info("Preparing inputs")
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

    
    is_async_mode = True   #CHG!!!

    ''' CHG '''
    wait_rate = rospy.Rate(1000)
    global rgb_img_update, depth_img_update, rgb_image, depth_img, bridge, motor_yaw_queue
    
    pub_total=rospy.Publisher("/yolo_ros_real_pose/detected_objects",ObjectsRealPose,queue_size = 1)
    corresponding_img_pub=rospy.Publisher("/yolo_ros_real_pose/img_for_detected_objects",Image,queue_size = 1)
    pub_time=rospy.Publisher("/yolo_ros_real_pose/detection_time", Float64, queue_size = 1)
    time_this = Float64()

    info_this = InfoGather()
    info_next = InfoGather()
    ''' END '''
    
    if args.input == "ROS":
        while not rospy.is_shutdown():
            if rgb_img_update == 1 and depth_img_update==1 and len(rgb_image)!=0 and len(depth_img)!=0 and (not motor_yaw_queue.empty() or not motor_yaw_syncronize):
                frame = rgb_image.copy()
                depth_this = depth_img.copy()

                info_this.uav_pose = uav_pose.pose
                info_this.img_to_pub = bridge.cv2_to_imgmsg(rgb_image, encoding="passthrough")
                info_this.img_to_pub.header = time_stamp_header

                if motor_yaw_syncronize:
                    info_this.motor_yaw = motor_yaw_queue.get()
                else:
                    info_this.motor_yaw = motor_yaw_corrected

                print("Time difference="+str(uav_pose.header.stamp.to_sec() - time_stamp_header.stamp.to_sec()))

                rgb_img_update = 0
                depth_img_update = 0
                wait_key_code = 1
                break
    else:
        input_stream = 0 if args.input == "cam" else args.input
        cap = cv2.VideoCapture(input_stream)
        number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

        wait_key_code = 1

        # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
        if number_input_frames != 1:
            ret, frame = cap.read()
        else:
            is_async_mode = False
            wait_key_code = 0

    # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between sync/async modes, press TAB key in the output window")
    
    while not rospy.is_shutdown():
        # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
        # in the regular mode, we capture frame to the CURRENT infer request
        try:
            detection_start_time = time()

            if args.input == "ROS":
                while not rospy.is_shutdown():
                    if rgb_img_update == 1 and depth_img_update==1 and (not motor_yaw_queue.empty() or not motor_yaw_syncronize):
                        if len(rgb_image)!=0 and len(depth_img)!=0:
                            if is_async_mode:
                                next_frame = rgb_image.copy()
                                depth_next = depth_img.copy()
                                info_next.uav_pose = uav_pose.pose
                                info_next.img_to_pub = bridge.cv2_to_imgmsg(rgb_image, encoding="passthrough")
                                info_next.img_to_pub.header = time_stamp_header

                                if motor_yaw_syncronize:
                                    info_next.motor_yaw = motor_yaw_queue.get()
                                else:
                                    info_next.motor_yaw = motor_yaw_corrected
                                
                            else:
                                frame = rgb_image.copy()
                                depth_this = depth_img.copy()
                                info_this.uav_pose = uav_pose.pose
                                info_this.img_to_pub = bridge.cv2_to_imgmsg(rgb_image, encoding="passthrough")
                                info_this.img_to_pub.header = time_stamp_header

                                if motor_yaw_syncronize:
                                    info_this.motor_yaw = motor_yaw_queue.get()
                                else:
                                    info_this.motor_yaw = motor_yaw_corrected


                            print("Time difference="+str(uav_pose.header.stamp.to_sec() - time_stamp_header.stamp.to_sec()))
                            rgb_img_update = 0
                            depth_img_update = 0
                            break

                    wait_rate.sleep()


            else:
                if is_async_mode:
                    ret, next_frame = cap.read()
                else:
                    ret, frame = cap.read()

                if not ret:
                    break


            if is_async_mode:
                request_id = next_request_id
                in_frame = cv2.resize(next_frame, (w, h))
            else:
                request_id = cur_request_id
                in_frame = cv2.resize(frame, (w, h))

            # resize input_frame to network size
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))

            # Start inference
            start_time = time()
            exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})
            det_time = time() - start_time

            # Collecting object detection results
            objects = list()
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                output = exec_net.requests[cur_request_id].outputs

                start_time = time()
                for layer_name, out_blob in output.items():
                    out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].shape)
                    layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
                    log.info("Layer {} parameters: ".format(layer_name))
                    layer_params.log_params()
                    objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                                 frame.shape[:-1], layer_params,
                                                 args.prob_threshold)
                parsing_time = time() - start_time

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

            if len(objects) and args.raw_output_message:
                log.info("\nDetected boxes for batch {}:".format(1))
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            origin_im_size = frame.shape[:-1]
            current_total_dect = ObjectsRealPose()

            for obj in objects:
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                         min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                if args.raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                                  obj['ymin'], obj['xmax'], obj['ymax'],
                                                                                  color))

                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                cv2.putText(frame,
                            "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

                '''CHG'''
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
                info.head_yaw = info_this.motor_yaw
                info.local_pose = info_this.uav_pose

                ### Now calculate the real position
                if len(depth_this)!=0 and (obj['xmax']-obj['xmin'])*(obj['ymax']-obj['ymin'])>=0:
                    ### Calculate position here by depth image and camera parameters
                    depth_box_width=obj['xmax']-obj['xmin']
                    depth_box_height=obj['ymax']-obj['ymin']
                    delta_rate=0.1
                    x_box_min=int(obj['xmin'] + depth_box_width*delta_rate)
                    y_box_min=int(obj['ymin'] + depth_box_height*delta_rate)
                    x_box_max=int(obj['xmax'] - depth_box_width*delta_rate)
                    y_box_max=int(obj['ymax'] - depth_box_height*delta_rate)
                    after_width=(depth_box_width*(1-2*delta_rate))
                    after_height=(depth_box_height*(1-2*delta_rate))
                    '''  '''
                    # rect = depth_this[y_box_min:y_box_max, x_box_min:x_box_max] # Simulation
                    rect = depth_this[y_box_min:y_box_max, x_box_min:x_box_max] * 0.001

                    rect[np.where(rect == 0)] = 99
                    rect[np.where(rect != rect)] = 99

                    z_pos = rect.min()

                    x_pos = (0.5 * (obj['xmax'] + obj['xmin']) - cx) * z_pos / fx
                    y_pos = (0.5 * (obj['ymax'] + obj['ymin']) - cy) * z_pos / fy

                    info.x = x_pos
                    info.y = y_pos
                    info.z = z_pos

                    current_total_dect.result.append(info)

            if len(current_total_dect.result)>0:
                current_total_dect.header = info_this.img_to_pub.header
                pub_total.publish(current_total_dect)
                corresponding_img_pub.publish(info_this.img_to_pub)

                time_this.data = time() - detection_start_time
                pub_time.publish(time_this)
                '''END'''

            # Draw performance stats over frame
            inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time * 1e3)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
            async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
                "Async mode is off. Processing request {}".format(cur_request_id)
            parsing_message = "YOLO parsing time is {:.3f}".format(parsing_time * 1e3)

            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(origin_im_size[0] - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(frame, parsing_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

            start_time = time()
            cv2.imshow("DetectionResults", frame)
            # cv2.imshow("depth_this", depth_this)
            render_time = time() - start_time

            ''' For next '''
            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame
                depth_this = depth_next
                info_this.uav_pose = info_next.uav_pose
                info_this.motor_yaw = info_next.motor_yaw
                info_this.img_to_pub = info_next.img_to_pub
                info_this.img_to_pub.header = info_next.img_to_pub.header

            key = cv2.waitKey(wait_key_code)

            # ESC key
            if key == 27:
                break
            # Tab key
            if key == 9:
                exec_net.requests[cur_request_id].wait()
                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

        except:
            print("An error occuered!!!!!!!")

    cv2.destroyAllWindows()


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
        global depth_img,depth_img_update,time_stamp_header, bridge
        self.depth_img=bridge.imgmsg_to_cv2(msg, "32FC1")
        depth_img=self.depth_img.copy()
        time_stamp_header=msg.header
        depth_img_update=1
        #rospy.loginfo("depth received!!")

    ''' Real world '''
    def motor_callback(self, msg):
        global motor_yaw_corrected, motor_yaw_init, if_init_motor, motor_yaw_queue
        if if_init_motor:
            motor_yaw_init = msg.x
            if_init_motor = False
        else:
            motor_yaw_corrected = -msg.x + motor_yaw_init
            if motor_yaw_queue.full():
                motor_yaw_queue.get()
                motor_yaw_queue.put(motor_yaw_corrected)
            else:
                motor_yaw_queue.put(motor_yaw_corrected)

    def uav_pose_callback(self, msg):
        global uav_pose
        uav_pose = msg

    ''' For Simulation '''
    # def motor_callback_sim(self, msg):
    #     global motor_yaw_corrected, motor_yaw_init, if_init_motor, motor_yaw_queue
    #     if if_init_motor:
    #         motor_yaw_init = msg.process_value
    #         if_init_motor = False
    #     else:
    #         motor_yaw_corrected = msg.process_value - motor_yaw_init
    #         if motor_yaw_queue.full():
    #             motor_yaw_queue.get()
    #             motor_yaw_queue.put(motor_yaw_corrected)
    #         else:
    #             motor_yaw_queue.put(motor_yaw_corrected)

    # def uav_pose_callback_sim(self, msg):
    #     global uav_pose
    #     uav_pose.pose = msg
    #     uav_pose.header.stamp = rospy.Time.now()


if __name__ == '__main__':
    rospy.init_node('detection', anonymous=False,log_level=rospy.INFO)
    sub_callbacks = SubscriberClass()

    image_sub = rospy.Subscriber(rgb_image_topic, Image, sub_callbacks.color_callback)  
    image_depth_sub = rospy.Subscriber(depth_image_topic, Image, sub_callbacks.depth_callback)
    motor_sub = rospy.Subscriber(motor_pose_topic, Point32, sub_callbacks.motor_callback)  
    position_sub = rospy.Subscriber(position_topic, PoseStamped, sub_callbacks.uav_pose_callback)

    #Simulation
    # motor_sub = rospy.Subscriber(motor_pose_topic, JointControllerState, sub_callbacks.motor_callback_sim)  
    # position_sub = rospy.Subscriber(position_topic, Pose, sub_callbacks.uav_pose_callback_sim)

    ''' Wait a little moment to update motor_yaw_queue '''
    quene_wait_rate = rospy.Rate(100)
    counter = 0
    while counter < 10:
        counter += 1
        quene_wait_rate.sleep()


    print("ROS initialized")
    if motor_yaw_syncronize:
        print("motor_yaw_syncronizer opened! Will only work when motor_pose_topic is published!")

    sys.exit(main() or 0)
