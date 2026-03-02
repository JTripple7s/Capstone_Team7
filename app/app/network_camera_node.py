#!/usr/bin/env python3
# encoding: utf-8

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class NetworkCameraNode(Node):
    def __init__(self):
        super().__init__('network_camera_node')

        # Select where frames come from:
        # - network: read MJPEG/HTTP stream from stream_url
        # - local: read webcam/device using camera_index or local_device
        self.camera_source = self.declare_parameter('camera_source', 'network').value
        self.stream_url = self.declare_parameter(
            'stream_url',
            'http://host.docker.internal:8080/mjpeg'
        ).value
        self.camera_index = int(self.declare_parameter('camera_index', 0).value)
        self.local_device = self.declare_parameter('local_device', '').value
        self.capture_backend = self.declare_parameter('capture_backend', 'auto').value
        self.output_topic = self.declare_parameter('output_topic', 'image_raw').value
        self.fps = float(self.declare_parameter('fps', 20.0).value)

        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.bridge = CvBridge()
        self.capture = None

        timer_period = 1.0 / self.fps if self.fps > 0 else 0.05
        self.timer = self.create_timer(timer_period, self.publish_frame)

        self.get_logger().info(f'camera_source: {self.camera_source}')
        self.get_logger().info(f'camera_index: {self.camera_index}')
        self.get_logger().info(f'local_device: {self.local_device}')
        self.get_logger().info(f'capture_backend: {self.capture_backend}')
        self.get_logger().info(f'stream_url: {self.stream_url}')
        self.get_logger().info(f'output_topic: {self.output_topic}')

    def _capture_source(self):
        # Resolve the OpenCV source argument based on the selected mode.
        if str(self.camera_source).lower() == 'local':
            if self.local_device:
                return self.local_device
            return self.camera_index
        return self.stream_url

    def _capture_backend_flag(self):
        # Optional backend override for platform-specific capture stability.
        backend = str(self.capture_backend).lower()
        if backend == 'avfoundation':
            return cv2.CAP_AVFOUNDATION
        if backend == 'v4l2':
            return cv2.CAP_V4L2
        if backend == 'dshow':
            return cv2.CAP_DSHOW
        if backend == 'ffmpeg':
            return cv2.CAP_FFMPEG
        if backend == 'gstreamer':
            return cv2.CAP_GSTREAMER
        return cv2.CAP_ANY

    def _ensure_capture(self):
        # Keep an active capture object; recreate it when disconnected.
        if self.capture is not None and self.capture.isOpened():
            return True

        if self.capture is not None:
            self.capture.release()

        source = self._capture_source()
        backend = self._capture_backend_flag()
        if backend == cv2.CAP_ANY:
            self.capture = cv2.VideoCapture(source)
        else:
            self.capture = cv2.VideoCapture(source, backend)

        if not self.capture.isOpened():
            self.get_logger().warning(
                f'Unable to open camera source ({source}) with backend ({self.capture_backend}), retrying...'
            )
            return False

        return True

    def publish_frame(self):
        if not self._ensure_capture():
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            # Drop and recreate capture on bad reads (network hiccup/camera reset).
            self.get_logger().warning('Stream read failed, reconnecting...')
            self.capture.release()
            self.capture = None
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NetworkCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
