#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from conductor.msg import TargetObject  # 导入自定义消息类型
import cv2
import numpy as np
import math
import time

class CircleDetectorNode(Node):
    def __init__(self):
        super().__init__('circle_detector_node')
        self.publisher = self.create_publisher(TargetObject, 'detected_circle', 10)
        self.cap = cv2.VideoCapture(0)    # 从摄像头捕获图像
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            print("Error: Camera could not be opened.")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)

            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to capture image')
            else:
                # 获取当前帧的宽度和高度
                height, width = frame.shape[:2]
                print(f"Frame size: {int(width)} x {int(height)}")

        self.timer = self.create_timer(1/10, self.detect_circle)
        self.running = True  # 控制程序运行的标志

    def detect_circle(self):
        if not self.running:
            return  # 如果不再运行，直接返回
        start_time = time.time()  # 记录开始时间

        ret, frame = self.cap.read()
        # height, width = frame.shape[:2]
        if not ret:
            self.get_logger().warn('Failed to capture image')
            return

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

        # 循环遍历所有轮廓
        for contour in contours:
            # 计算轮廓的面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # 如果周长为零，则跳过此轮廓
            if perimeter == 0:
                continue
            # 计算近似轮廓并计算其与原始轮廓之间的差异
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            # 如果轮廓是圆形，则绘制它
            if len(approx) > 8 and circularity > 0.7 and area > 1000:
                # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                ellipse = cv2.fitEllipse(contour)
                # 获取椭圆参数
                center, axes, angle = ellipse
                x_center, y_center = int(center[0]), int(center[1])

                # 在图像上绘制中心点
                cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)
                # 在图像上绘制中心点的值
                cv2.putText(frame, f"({x_center}, {y_center})", (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                # 绘制椭圆和识别框
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
        # cv2.imshow("HSV", hsv)
        # end_time = time.time()  # 记录结束时间
        # elapsed_time = end_time - start_time  # 计算耗时
        # self.get_logger().info(f"Processing time: {elapsed_time:.4f} seconds")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False  # 设置运行标志为 False

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CircleDetectorNode()
    try:
        while node.running and rclpy.ok():
            # node.detect_circle()  # 手动调用检测函数
            rclpy.spin_once(node)
    finally:
        node.cleanup()  # 释放资源并关闭窗口
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

