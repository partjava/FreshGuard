# 🍎 鲜智巡检 —— AI赋能基层果蔬智能质检系统

一个基于深度学习的水果质量检测与追踪系统，支持自动识别水果种类和新鲜程度。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO11-Ultralytics-green.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-Academic%20Use%20Only-red.svg)](#许可证)

## 📋 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [功能特点](#功能特点)
- [技术栈](#技术栈)
- [安装部署](#安装部署)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [模型训练](#模型训练)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

**鲜智巡检**是一个基于深度学习的智能水果质量检测系统，专为基层果蔬质检场景设计。系统采用Jetson Nano移动小车作为执行端，搭载摄像头进行自主巡航和目标定位，通过局域网将画面传输至PC端进行实时推理分析。

### 核心价值

- 🎯 **精准检测**：基于YOLO11深度学习模型，识别准确率达95%+
- 🚗 **自主巡航**：小车自动寻找、定位水果，无需人工干预
- 🤖 **智能交互**：支持语音、文字、AI对话多种交互方式
- 📊 **质量判断**：自动区分新鲜/腐烂状态，辅助质检决策
- 💡 **本地部署**：全部AI推理在本地完成，保护数据隐私

### 应用场景

- 农产品批发市场质检
- 果蔬分拣生产线
- 智慧农业巡检
- 食品安全监管

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        系统架构图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │   Jetson Nano    │         │     PC 端        │             │
│  │   (移动端)       │  TCP    │    (推理端)      │             │
│  │                  │◄───────►│                  │             │
│  │  ┌────────────┐  │  Socket │  ┌────────────┐  │             │
│  │  │ 摄像头模块 │  │         │  │ 视频接收   │  │             │
│  │  └────────────┘  │         │  └────────────┘  │             │
│  │  ┌────────────┐  │         │  ┌────────────┐  │             │
│  │  │ 电机控制   │  │         │  │ YOLO推理   │  │             │
│  │  └────────────┘  │         │  └────────────┘  │             │
│  │  ┌────────────┐  │         │  ┌────────────┐  │             │
│  │  │ 指令接收   │  │         │  │ AI对话     │  │             │
│  │  └────────────┘  │         │  └────────────┘  │             │
│  └──────────────────┘         └──────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 工作流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  扫描   │───►│  追踪   │───►│  到达   │───►│  检测   │───►│  分析   │
│ 8个方向 │    │ 颜色+YOLO│    │  停车   │    │ 投票判断 │    │  AI回复 │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼
  旋转搜索      PID控制       稳定帧采集     5次检测投票    多模态分析
  颜色+YOLO     平滑转向      等待停稳       取最高置信度   语音播报
```

---

## 功能特点

### 🎯 智能检测

- **两阶段策略**：远距离颜色追踪 + 近距离YOLO精确识别
- **多帧投票**：到达后连续检测5帧，投票决定最终结果
- **置信度阈值**：可配置的置信度过滤，确保检测可靠性

### 🚗 自主导航

- **PID控制**：位置式PID控制器实现平滑转向
- **颜色追踪**：HSV颜色空间快速定位目标
- **记忆补偿**：短期记忆机制防止单帧丢失

### 🤖 AI交互

- **多模态对话**：支持图片+文字+语音的混合交互
- **实时分析**：到达水果后自动触发AI质量分析
- **语音合成**：AI回复自动朗读，支持离线使用

### 📊 数据管理

- **结果记录**：自动记录检测结果到CSV/TXT文件
- **历史查询**：支持按时间、水果类型查询检测记录
- **可视化**：检测结果实时显示在GUI界面

### 🎮 多种控制

- **GUI控制**：图形界面一键操作
- **手动控制**：方向按钮精细操控
- **语音控制**：按住说话语音指令
- **AI对话**：自然语言交互

---

## 技术栈

### 核心技术

| 技术 | 用途 | 版本 |
|------|------|------|
| YOLO11 | 目标检测 | Ultralytics |
| OpenCV | 计算机视觉 | 4.8+ |
| PyTorch | 深度学习框架 | 2.0+ |
| Ollama | 本地AI推理 | - |
| faster-whisper | 语音识别 | - |
| edge-tts | 语音合成 | - |
| Tkinter | GUI界面 | Python内置 |

### 硬件平台

| 设备 | 型号 | 用途 |
|------|------|------|
| 移动端 | Jetson Nano B01 | 摄像头采集、电机控制 |
| 推理端 | PC (RTX 4070 8G) | YOLO推理、AI分析 |
| 小车底盘 | JetMini Bot | 移动平台 |
| 摄像头 | USB/CSI | 图像采集 |

---

## 安装部署

### 1. 环境准备

#### PC端

```bash
# 创建虚拟环境
python -m venv fruit_env
source fruit_env/bin/activate  # Linux/Mac
# fruit_env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

#### Jetson Nano端

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y python3-pip

# 安装Jetson依赖
pip3 install jetbotmini
```

### 2. 配置网络

编辑 `src/config.py`，修改IP地址：

```python
# 网络配置
CAR_IP = "192.168.1.4"    # Jetson Nano IP
PC_IP = "192.168.1.10"     # 电脑IP
VIDEO_PORT = 8080          # 画面接收端口
CMD_PORT = 8081            # 指令发送端口
```

**注意：** 确保Jetson Nano和PC在同一局域网内！

### 3. 启动系统

#### 启动Jetson Nano端

```bash
cd src/jetson
python nano_main.py
```

#### 启动PC端

```bash
cd src/pc
python main_pc.py
```

---

## 使用说明

### 基本操作

1. **启动系统**
   - 先启动Jetson Nano端程序
   - 再启动PC端程序
   - 等待摄像头连接成功

2. **选择水果**
   - 在GUI界面输入水果名称（如：apples, banana, oranges）
   - 支持的水果：apple, banana, orange, grape, mango

3. **开始追踪**
   - 点击"▶ 开始追踪"按钮
   - 小车将自动扫描并追踪目标水果

4. **查看结果**
   - 到达水果后自动检测并显示结果
   - ✅ 新鲜 (Fresh) / ❌ 腐烂 (Rotten)
   - 置信度百分比

5. **AI分析**
   - 到达后自动触发AI分析
   - 可在输入框追问相关问题
   - 支持语音输入

### 手动控制

使用方向按钮控制小车：
- `↖` `↑` `↗` - 左前、前进、右前
- `←` `⏹` `→` - 左转、停止、右转
- `↙` `↓` `↘` - 左后、后退、右后

**按住按钮移动，松开停止**

### 语音交互

1. 点击"🎤 按住说话"按钮
2. 说出你的问题
3. 松开按钮，等待识别
4. 识别结果自动填入输入框
5. 按回车或点击"发送"

### AI对话示例

- "这个苹果新鲜吗？"
- "帮我分析一下图片中的水果"
- "什么水果适合做沙拉？"
- "如何判断香蕉是否成熟？"

---

## 项目结构

```
challenge_cup_fruit/
│
├── src/                          # 源代码目录
│   ├── pc/                       # PC端程序
│   │   ├── main_pc.py           # PC端主程序（GUI界面）
│   │   └── test_detect.py       # 检测测试脚本
│   │
│   ├── jetson/                   # Jetson Nano端程序
│   │   ├── nano_main.py         # Nano端主程序
│   │   ├── nano_camera.py       # 摄像头模块
│   │   ├── nano_motor.py        # 电机控制模块
│   │   └── config.py            # Nano端配置
│   │
│   ├── config.py                 # 全局配置文件
│   ├── detector.py               # YOLO检测器
│   ├── fruit_controller.py       # 水果追踪控制器
│   ├── commander.py              # 指令发送模块
│   ├── video_receiver.py         # 视频接收模块
│   ├── pid.py                    # PID控制器
│   ├── ai_chat.py                # AI对话模块
│   ├── calibrate_color.py        # 颜色校准工具
│   └── generate_fruit_labels.py  # 标签生成工具
│
├── models/                       # 训练好的模型
│   ├── fruit_quality.pt          # 基础模型
│   ├── fruit_quality_enhanced.pt # 增强模型
│   └── fruit_quality_max.pt      # 最优模型
│
├── dataset/                      # 训练数据集
│   ├── data.yaml                 # 数据集配置
│   ├── images/                   # 图片数据
│   │   ├── train/               # 训练集（54,584张）
│   │   └── val/                 # 验证集（6,658张）
│   └── labels/                   # 标注数据
│
├── runs/                         # 训练日志
│   └── fruit_quality_max3/       # 最新训练结果
│       ├── weights/             # 模型权重
│       ├── results.csv          # 训练曲线
│       └── confusion_matrix.png # 混淆矩阵
│
├── train_yolo_fruit_enhanced.py  # 训练脚本
├── train_yolo.py                 # 基础训练脚本
├── md_to_docx.py                 # 文档转换工具
│
├── *.pptx                        # 演示文稿
├── *.docx                        # 项目文档
├── *.pdf                         # PDF文档
│
└── README.md                     # 本说明文档
```

---

## 模型训练

### 训练数据

- **数据集**：5种水果 × 2种状态 = 10个类别
- **训练集**：54,584 张图片
- **验证集**：6,658 张图片
- **标注格式**：YOLO TXT格式

### 类别定义

```yaml
# dataset/data.yaml
names:
  - apple_fresh      # 新鲜苹果
  - apple_rotten     # 腐烂苹果
  - banana_fresh     # 新鲜香蕉
  - banana_rotten    # 腐烂香蕉
  - grape_fresh      # 新鲜葡萄
  - grape_rotten     # 腐烂葡萄
  - mango_fresh      # 新鲜芒果
  - mango_rotten     # 腐烂芒果
  - orange_fresh     # 新鲜橙子
  - orange_rotten    # 腐烂橙子
nc: 10
```

### 训练命令

```bash
# 使用增强版训练脚本（推荐）
python train_yolo_fruit_enhanced.py

# 或使用基础训练脚本
python train_yolo.py
```

### 训练参数

```python
# 核心参数（RTX 4070 8G优化版）
model = YOLO('yolo11m.pt')
results = model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,           # 4070 8G显存安全值
    device=0,
    patience=20,       # 早停机制
    
    # 数据增强
    degrees=180,       # 随机旋转
    translate=0.3,     # 平移
    scale=0.6,         # 缩放
    mosaic=1.0,        # 马赛克增强
)
```

### 训练结果

最新训练（fruit_quality_max3）：
- **mAP@0.5**：95.35%
- **mAP@0.5-95**：92.48%
- **Precision**：89.54%
- **Recall**：93.46%
- **训练时间**：约19.3小时

---

## 配置说明

### 网络配置

编辑 `src/config.py`：

```python
# 网络配置
CAR_IP = "192.168.1.4"    # Jetson Nano IP地址
PC_IP = "192.168.1.10"     # PC端IP地址
VIDEO_PORT = 8080          # 视频传输端口
CMD_PORT = 8081            # 指令传输端口

# YOLO识别配置
CONF_THRESHOLD = 0.50      # 置信度阈值
```

### 追踪参数

编辑 `src/fruit_controller.py`：

```python
# 追踪参数
CENTER_DEAD_ZONE = 30      # 转向死区（像素）
ARRIVE_AREA = 0.08         # 到达面积阈值
LOST_FRAMES = 5            # 丢失帧数阈值

# 速度参数
SCAN_TURN_SPEED = 0.25     # 扫描旋转速度
FOLLOW_SPEED = 0.45        # 追踪速度
SLOW_SPEED = 0.25          # 接近时减速
TURN_GAIN = 0.6            # 转向增益
```

### 颜色范围

编辑 `src/fruit_controller.py` 中的 `COLOR_RANGES`：

```python
COLOR_RANGES = {
    'oranges': [(np.array([5, 120, 100]), np.array([20, 255, 255]))],
    'banana':  [(np.array([20, 100, 120]), np.array([35, 255, 255]))],
    'apples':  [
        (np.array([0, 60, 100]), np.array([10, 255, 255])),
        (np.array([165, 60, 100]), np.array([179, 255, 255]))
    ],
    # ... 其他水果
}
```

**提示：** 使用 `calibrate_color.py` 工具校准颜色范围

---

## 常见问题

### Q1: 摄像头无法连接

**检查项：**
- Jetson Nano和PC是否在同一局域网
- IP地址是否正确配置
- 防火墙是否放行端口8080/8081
- 摄像头是否正常工作

**解决方案：**
```bash
# 测试网络连通性
ping 192.168.1.4  # Jetson Nano IP

# 检查端口占用
netstat -an | grep 8080
```

### Q2: YOLO检测不准确

**可能原因：**
- 光线条件不佳
- 水果角度特殊
- 模型需要重新训练

**解决方案：**
1. 调整置信度阈值 `CONF_THRESHOLD`
2. 使用 `calibrate_color.py` 校准颜色
3. 增加训练数据重新训练模型

### Q3: 小车运动不稳定

**调整参数：**
```python
# 在 fruit_controller.py 中调整
CENTER_DEAD_ZONE = 40  # 增大死区，减少抖动
TURN_GAIN = 0.4        # 减小转向增益
FOLLOW_SPEED = 0.3     # 降低速度
```

### Q4: AI响应慢

**优化方案：**
1. 确保Ollama服务已启动
2. 使用更小的模型（如gemma3:4b）
3. 检查GPU内存使用情况
4. 减少图片分辨率

### Q5: 训练显存不足

**解决方案：**
```python
# 减小batch size
batch=4  # 或更小

# 使用更小的模型
model = YOLO('yolo11s.pt')  # small版本

# 减小输入尺寸
imgsz=480
```

---

## 性能指标

### 检测性能

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| mAP@0.5 | ≥90% | 95.35% | ✅ |
| mAP@0.5-95 | ≥85% | 92.48% | ✅ |
| Precision | ≥85% | 89.54% | ✅ |
| Recall | ≥85% | 93.46% | ✅ |

### 系统性能

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 推理速度 | ≥10fps | 15fps | ✅ |
| 追踪响应 | ≤100ms | 50ms | ✅ |
| 系统稳定性 | ≥1小时 | 持续运行 | ✅ |
| 网络延迟 | ≤100ms | 30ms | ✅ |

### 资源占用

| 资源 | PC端 | Jetson Nano |
|------|------|-------------|
| CPU | 15% | 45% |
| GPU | 60% | - |
| 内存 | 2GB | 1.5GB |
| 显存 | 4GB | - |

---

## 许可证

本项目采用 **学术研究许可证**，具体条款如下：

```
Copyright (c) 2024 鲜智巡检团队

学术研究许可协议

特此授予任何获得本软件副本的人免费使用权，但仅限于以下目的：
1. 学术研究
2. 教育教学
3. 个人学习
4. 非商业性实验

禁止以下用途：
1. 商业使用（包括但不限于销售、出租、盈利性服务）
2. 将本软件用于任何商业产品或服务
3. 未经授权的分发或传播
4. 修改后闭源发布

以上限制同样适用于衍生作品。

本软件按"原样"提供，作者不对任何因使用本软件而导致的损失负责。
使用本软件即表示您同意以上条款。

如需商业使用，请联系项目团队获取授权。
```

---

## 🙏 致谢

感谢以下开源项目和技术：

- [Ultralytics YOLO](https://ultralytics.com) - 目标检测框架
- [Ollama](https://ollama.com) - 本地AI推理
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - 语音识别
- [edge-tts](https://github.com/rany2/edge-tts) - 语音合成
- [Jetson Nano](https://developer.nvidia.com/embedded-computing) - 嵌入式AI平台

---

**最后更新：2024年3月**
