# Jetson Nano 端代码

这个文件夹包含所有运行在 Jetson Nano 小车上的代码。

## 文件说明

- `config.py` - 配置文件（网络IP、端口、摄像头参数）
- `nano_camera.py` - 摄像头控制模块
- `nano_motor.py` - 电机控制模块
- `nano_main.py` - 主程序（启动这个文件）

## 部署到 Jetson Nano

### 1. 复制文件到 Jetson Nano

```bash
# 在 Jetson Nano 上创建目录
mkdir -p ~/challenge_cup_fruit/src

# 从 Windows PC 复制文件（使用 scp 或 U盘）
# 方法1：使用 scp（需要知道 Jetson 的 IP）
scp src/jetson/* liming@192.168.1.15:~/challenge_cup_fruit/src/

# 方法2：使用 U盘
# 把 src/jetson/ 下的所有文件复制到 U盘
# 插到 Jetson Nano 上，复制到 ~/challenge_cup_fruit/src/
```

### 2. 修改 IP 地址

编辑 `config.py`，修改默认 IP：

```python
CAR_IP = os.getenv("CAR_IP", "你的Jetson IP")
PC_IP = os.getenv("PC_IP", "你的电脑IP")
```

或者每次启动时设置环境变量：

```bash
export CAR_IP=192.168.1.15
export PC_IP=192.168.1.10
python nano_main.py
```

### 3. 查看 IP 地址

```bash
# Jetson Nano 上查看自己的 IP
ifconfig wlan0  # WiFi
# 或
ifconfig eth0   # 网线

# Windows PC 上查看自己的 IP
ipconfig
```

### 4. 测试模块

```bash
cd ~/challenge_cup_fruit/src

# 测试摄像头
python nano_camera.py

# 测试电机
python nano_motor.py
```

### 5. 运行主程序

```bash
python nano_main.py
```

## 常见问题

### Q: IP 地址经常变化怎么办？

A: 创建启动脚本 `start.sh`：

```bash
#!/bin/bash
export CAR_IP=192.168.1.15
export PC_IP=192.168.1.10
cd ~/challenge_cup_fruit/src
python nano_main.py
```

然后：
```bash
chmod +x start.sh
./start.sh
```

### Q: 摄像头连接失败？

A: 检查：
1. PC 端的 `video_receiver.py` 是否已启动
2. IP 地址是否正确
3. 防火墙是否阻止了端口 8080

### Q: 电机不动？

A: 检查：
1. 电池是否有电
2. 电机驱动板是否正常
3. 运行 `python nano_motor.py` 单独测试

### Q: 如何停止程序？

A: 按 `Ctrl+C`
