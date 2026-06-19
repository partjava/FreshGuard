"""
指令发送模块
功能：连接 Jetson Nano 指令端口，发送运动控制指令
"""
import socket
import threading
import time
from config import CAR_IP, CMD_PORT


class Commander:
    def __init__(self):
        self._socket = None
        self._thread = threading.Thread(target=self._connect_loop, daemon=True)

    def start(self):
        """启动后台连接线程"""
        self._thread.start()

    def send(self, cmd: str):
        """发送指令（16字节固定长度，与 Jetson 端协议一致）"""
        if self._socket is None:
            return
        try:
            self._socket.sendall(cmd.ljust(32).encode('utf-8'))
        except Exception as e:
            print(f"❌ 指令发送失败：{e}")
            self._socket = None

    def _connect_loop(self):
        """后台线程：持续尝试连接 Jetson 指令端口"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                s.connect((CAR_IP, CMD_PORT))
                self._socket = s
                print(f"✅ 指令通道已连接 {CAR_IP}:{CMD_PORT}")
                break
            except Exception as e:
                print(f"🔄 等待 Jetson 指令端口... {e}")
                time.sleep(1)
