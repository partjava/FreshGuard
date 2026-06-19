"""
Jetbotmini 主程序
作用：整合摄像头+电机+指令监听，接收电脑指令并执行
"""
import socket
import threading
import time
from nano_camera import NanoCamera
from nano_motor import NanoMotor
from config import CMD_PORT, FORWARD_SPEED, TURN_SPEED, BACKWARD_SPEED

# 初始化核心模块
motor = NanoMotor()
camera = NanoCamera()


def listen_command():
    """监听电脑发来的控制指令，执行对应运动"""
    cmd_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd_server.bind(("0.0.0.0", CMD_PORT))
    cmd_server.listen(1)
    print(f"🔍 小车监听指令端口 {CMD_PORT}...")
    
    conn, addr = cmd_server.accept()
    print(f"💻 电脑 {addr} 已连接指令端口")
    
    try:
        while True:
            cmd = conn.recv(32).decode('utf-8').strip()
            if not cmd:
                break
            
            if cmd.startswith("motors:"):
                # PID连续控制：motors:左轮速度:右轮速度
                _, sl, sr = cmd.split(":")
                motor.robot.set_motors(float(sl), float(sr))
            elif cmd == "forward":
                motor.forward(FORWARD_SPEED)
            elif cmd == "left":
                motor.left(TURN_SPEED)
            elif cmd == "right":
                motor.right(TURN_SPEED)
            elif cmd == "backward":
                motor.backward(BACKWARD_SPEED)
            elif cmd == "stop":
                motor.stop()
    except Exception as e:
        print(f"❌ 指令监听出错：{e}")
        motor.stop()
    finally:
        conn.close()
        cmd_server.close()
        motor.stop()


if __name__ == "__main__":
    try:
        # 启动指令监听线程
        cmd_thread = threading.Thread(target=listen_command, daemon=True)
        cmd_thread.start()
        time.sleep(1)
        
        # 连接电脑并持续发送画面
        if camera.connect_to_pc():
            while True:
                camera.send_frame()
        else:
            print("❌ 无法连接到电脑，程序退出")
    except KeyboardInterrupt:
        motor.stop()
        camera.close()
        print("🛑 小车程序已手动停止")
    except Exception as e:
        motor.stop()
        camera.close()
        print(f"❌ 程序异常退出：{e}")
