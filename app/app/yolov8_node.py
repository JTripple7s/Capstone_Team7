#!/usr/bin/env python3
import cv2
import time
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from interfaces.msg import ObjectInfo, ObjectsInfo

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')

        # Parameters
        self.declare_parameter('model_path', 'app/models/soccer_v8.onnx')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 640)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        self.input_width = self.get_parameter('input_width').get_parameter_value().integer_value
        self.input_height = self.get_parameter('input_height').get_parameter_value().integer_value

        # YOLOv8 Class Names - Updated for single-class ball detection
        # If your model has only 1 class, this list should match.
        self.class_names = ['ball'] 

        # Load ONNX Model
        try:
            # Note: providers=['CPUExecutionProvider'] is best for Raspberry Pi 4/5
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.get_logger().info(f"Loaded YOLOv8 model from {model_path}")
            
            # Print model input/output info for debugging
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            self.get_logger().info(f"Model Input: {inputs[0].name}, Shape: {inputs[0].shape}")
            self.get_logger().info(f"Model Output: {outputs[0].name}, Shape: {outputs[0].shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {str(e)}")
            return

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(ObjectsInfo, '/soccer/detections', 10)
        self.img_pub = self.create_publisher(Image, '/soccer/debug_image', 10)

    def preprocess(self, img):
        # YOLOv8 expects 640x640 usually. Using linear interpolation for speed.
        input_img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1) # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0) # CHW to BCHW
        return input_img

    def postprocess(self, outputs, original_img):
        img_h, img_w = original_img.shape[:2]
        output = outputs[0][0] # (4 + num_classes, 8400)
        output = output.transpose() # (8400, 4 + num_classes)

        boxes = []
        confidences = []
        class_ids = []

        # Optimization: Filter by confidence before expensive box calculations
        # YOLOv8 output: [x_center, y_center, width, height, class0_score, class1_score, ...]
        for row in output:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence > self.conf_threshold:
                x, y, w, h = row[0:4]
                
                # Scale from 640x640 back to original image size
                x_scale = img_w / self.input_width
                y_scale = img_h / self.input_height
                
                x1 = int((x - w/2) * x_scale)
                y1 = int((y - h/2) * y_scale)
                bw = int(w * x_scale)
                bh = int(h * y_scale)

                boxes.append([x1, y1, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # NMS to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        objects_msg = ObjectsInfo()
        if len(indices) > 0:
            for i in indices: # cv2.dnn.NMSBoxes returns a flat list in newer versions
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                box = boxes[idx]
                obj = ObjectInfo()
                
                # Safeguard for class index out of range
                c_id = class_ids[idx]
                obj.class_name = self.class_names[c_id] if c_id < len(self.class_names) else f"class_{c_id}"
                
                obj.box = box # [x, y, w, h]
                obj.score = confidences[idx]
                obj.width = img_w
                obj.height = img_h
                objects_msg.objects.append(obj)
                
                # Draw on original image for debug
                color = (0, 255, 0) # Green for ball
                cv2.rectangle(original_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                label = f"{obj.class_name}: {obj.score:.2f}"
                cv2.putText(original_img, label, (box[0], box[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return objects_msg, original_img

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        start_time = time.time()
        input_data = self.preprocess(cv_img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
        objects_msg, debug_img = self.postprocess(outputs, cv_img)
        fps = 1.0 / (time.time() - start_time)

        # Publish detections
        self.pub.publish(objects_msg)

        # Add FPS to debug image and publish
        cv2.putText(debug_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
