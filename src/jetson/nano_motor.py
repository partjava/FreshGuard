"""
Jetbotmini 电机控制模块
作用：封装官方Robot类，提供标准化运动接口
"""
from jetbotmini import Robot


class NanoMotor:
    def __init__(self):
        # 初始化Jetbotmini电机驱动
        self.robot = Robot()
        print("✅ 电机初始化完成")

    def forward(self, speed=1.0):
        """前进"""
        self.robot.forward(speed)
        print(f"🚗 前进（速度：{speed}）")

    def left(self, speed=0.75):
        """左转"""
        self.robot.left(speed)
        print(f"🚗 左转（速度：{speed}）")

    def right(self, speed=0.75):
        """右转"""
        self.robot.right(speed)
        print(f"🚗 右转（速度：{speed}）")

    def backward(self, speed=1.0):
        """后退"""
        self.robot.backward(speed)
        print(f"🚗 后退（速度：{speed}）")

    def stop(self):
        """停止（必须调用，防止电机发烫）"""
        self.robot.stop()
        print("🛑 电机已停止")


# 单独测试电机
if __name__ == "__main__":
    motor = NanoMotor()
    import time
    try:
        # 测试前进
        motor.forward(1.0)
        time.sleep(2)
        motor.stop()
        time.sleep(1)
        
        # 测试左转
        motor.left(0.75)
        time.sleep(2)
        motor.stop()
        time.sleep(1)
        
        # 测试右转
        motor.right(0.75)
        time.sleep(2)
        motor.stop()
    finally:
        motor.stop()
        print("✅ 电机测试结束")
