#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from conductor.msg import TargetObject  # 导入自定义消息类型
import cv2
import numpy as np
import math
import time
import threading

class CircleDetectorNode(Node):
    def __init__(self):
        super().__init__('circle_detector_node')
        self.publisher = self.create_publisher(TargetObject, 'detected_circle', 10)
        self.cap = cv2.VideoCapture(0)  # 从摄像头捕获图像
        if not self.cap.isOpened():
            print("Error: Camera could not be opened.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.running = True  # 控制程序运行的标志
        self.thread = threading.Thread(target=self.capture_and_process)
        self.thread.start()

    def capture_and_process(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to capture image')
                continue

            # 直接调用图像处理方法
            self.detect_circle(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False  # 设置运行标志为 False

    def detect_circle(self, frame):
        start_time = time.time()  # 记录开始时间

        # 将图像转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义红色范围
        lower_red1 = np.array([0, 178, 118])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 114, 77])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # 将两个掩码组合在一起
        mask = mask1 + mask2

        # 对掩码进行形态学操作以消除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=4)

        # 查找图像中的所有轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if len(approx) > 8 and circularity > 0.7 and area > 1000:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                x_center, y_center = int(center[0]), int(center[1])

                # 在图像上绘制中心点
                cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"({x_center}, {y_center})", (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                # 创建并发布消息
                msg = TargetObject()
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'red'
                msg.center = Point(x=float(center[0]), y=float(center[1]), z=0.0)

                self.publisher.publish(msg)
                self.get_logger().info(f"Published: ({x_center}, {y_center})")

        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)

    def cleanup(self):
        self.running = False
        self.thread.join()  # 等待线程结束
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetectorNode()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        while node.running and rclpy.ok():
            executor.spin_once()
    finally:
        node.cleanup()  # 释放资源并关闭窗口
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
