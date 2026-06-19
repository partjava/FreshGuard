"""
Jetbotmini 摄像头模块（优化版）
作用：初始化官方摄像头，向电脑发送画面
改进：增加错误处理、连接重试、帧率控制
"""
import socket
import cv2
import numpy as np
import time
from jetbotmini import Camera
from config import PC_IP, VIDEO_PORT, CAMERA_WIDTH, CAMERA_HEIGHT, JPEG_QUALITY


class NanoCamera:
    def __init__(self):
        # 初始化Jetbotmini官方摄像头
        self.camera = Camera.instance(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        print(f"✅ 摄像头初始化完成 ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
        
        # 初始化socket
        self.video_socket = None
        self.connected = False

    def connect_to_pc(self, max_retries=10):
        """连接电脑的视频接收端口，失败自动重试"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.video_socket.connect((PC_IP, VIDEO_PORT))
                self.connected = True
                print(f"✅ 摄像头已连接电脑 {PC_IP}:{VIDEO_PORT}")
                return True
            except Exception as e:
                retry_count += 1
                print(f"🔄 重试连接电脑 ({retry_count}/{max_retries})... 错误：{e}")
                time.sleep(2)
        
        print(f"❌ 无法连接到电脑 {PC_IP}:{VIDEO_PORT}")
        return False

    def send_frame(self):
        """发送单帧画面到电脑（压缩为JPG，减少传输量）"""
        if not self.connected:
            return False
        
        try:
            # 获取摄像头画面
            frame = self.camera.value
            
            # 压缩为JPG格式
            _, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            data = np.array(img_encode).tobytes()
            
            # 先发送数据长度（防止粘包），再发送画面数据
            self.video_socket.sendall(str(len(data)).ljust(16).encode('utf-8'))
            self.video_socket.sendall(data)
            return True
        except Exception as e:
            print(f"❌ 发送画面失败：{e}")
            self.connected = False
            return False

    def close(self):
        """安全关闭摄像头和socket"""
        try:
            if self.video_socket:
                self.video_socket.close()
            del self.camera
            print("🛑 摄像头已关闭")
        except Exception as e:
            print(f"⚠️ 关闭摄像头时出错：{e}")


# 单独测试摄像头
if __name__ == "__main__":
    camera = NanoCamera()
    if camera.connect_to_pc():
        try:
            frame_count = 0
            start_time = time.time()
            while True:
                if camera.send_frame():
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"📊 已发送 {frame_count} 帧，平均帧率：{fps:.1f} fps")
                else:
                    print("⚠️ 发送失败，尝试重连...")
                    if not camera.connect_to_pc():
                        break
        except KeyboardInterrupt:
            camera.close()
            print("✅ 摄像头测试结束")
