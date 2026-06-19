"""
视频接收模块
功能：监听端口，接收 Jetson Nano 发来的 JPEG 画面帧
"""
import socket
import threading
import cv2
import numpy as np
from config import PC_IP, VIDEO_PORT


class VideoReceiver:
    def __init__(self):
        self.latest_frame = None
        self._lock = threading.Lock()
        self._on_connected = None  # 连接成功回调
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)

    def start(self):
        """启动后台接收线程"""
        self._thread.start()

    def get_frame(self):
        """获取最新一帧（线程安全）"""
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _receive_loop(self):
        """后台线程：持续接收 Jetson 发来的画面"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((PC_IP, VIDEO_PORT))
        server.listen(1)
        print(f"📷 等待 Jetson 摄像头连接 {PC_IP}:{VIDEO_PORT} ...")
        conn, addr = server.accept()
        print(f"✅ 摄像头已连接：{addr}")
        # 摄像头连接后触发回调（用于预热AI模型）
        if hasattr(self, '_on_connected') and self._on_connected:
            threading.Thread(target=self._on_connected, daemon=True).start()

        while True:
            try:
                # 先读 16 字节长度头，再读完整图片数据
                header = self._recv_exact(conn, 16)
                data_len = int(header.decode('utf-8').strip())
                data = self._recv_exact(conn, data_len)

                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self.latest_frame = frame
            except Exception as e:
                print(f"❌ 视频接收出错：{e}")
                break

    @staticmethod
    def _recv_exact(conn, n: int) -> bytes:
        """确保接收到恰好 n 字节，防止粘包"""
        buf = b''
        while len(buf) < n:
            chunk = conn.recv(min(4096, n - len(buf)))
            if not chunk:
                raise ConnectionResetError("连接断开")
            buf += chunk
        return buf
