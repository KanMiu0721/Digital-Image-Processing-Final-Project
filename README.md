# 基于手势识别的机械狗控制系统

> 期末项目 —— 通过摄像头识别手势，MQTT 协议远程控制机械狗做出对应动作

## 项目简介

本系统使用笔记本摄像头捕获实时视频，基于 **MediaPipe Hands** 提取手部 21 个关键点，通过自定义算法识别三种手势，经由 **MQTT 协议** 发送控制指令，最终由 **可视化模拟器**（或真实机械狗）执行对应动作。

无需树莓派、无需机械狗硬件，纯软件环境即可完整运行和演示。

---

## 系统架构

```
摄像头 ──► MediaPipe Hands ──► 手势检测器 ──► MQTT 发布 ──► MQTT Broker ──► 模拟器/机械狗
  (OpenCV)   (21关键点)      (GestureDetector)  (paho-mqtt)    (mosquitto)     (dog_simulator)
```

| 层级   | 文件                  | 职责                                          |
| ------ | --------------------- | --------------------------------------------- |
| 感知层 | `main.py`             | 摄像头采集、MediaPipe 推理、可视化界面        |
| 算法层 | `gesture_detector.py` | 手型判定（平摊/握拳）、垂直运动检测、画圈检测 |
| 通信层 | `mqtt_publisher.py`   | MQTT 指令发布（JSON 格式）                    |
| 配置层 | `config.py`           | 所有可调参数集中管理                          |
| 执行层 | `dog_simulator.py`    | 机械狗可视化模拟器，接收 MQTT 并执行动画      |

---

## 手势 → 动作对照

| 手势       | 手型       | 运动           | 指令     | 狗的动作     |
| ---------- | ---------- | -------------- | -------- | ------------ |
| 平摊向下挥 | 四指伸直 ✋ | 手腕快速下降 ↓ | `SIT`    | 站立 → 坐下  |
| 平摊向上挥 | 四指伸直 ✋ | 手腕快速上升 ↑ | `STAND`  | 坐下 → 站起  |
| 握拳画圈   | 四指全弯 👊 | 手腕画圆 ○     | `ROTATE` | 原地转身一圈 |

### 检测算法

| 手势     | 方法                                                                 |
| -------- | -------------------------------------------------------------------- |
| 手型判定 | **距离比值法**：手指伸直时 `指尖→指根直线距离 / 指节路径总长 > 0.75` |
| 上下挥   | **滑动窗口 + Savitzky-Golay 滤波**：5 帧窗口内 Y 轴位移和速度超阈值  |
| 握拳画圈 | **累计转角法**：15 帧窗口内手腕轨迹累计转角 > 330° 且轨迹半径 > 阈值 |

---

## 环境要求

- Python 3.10+
- 摄像头（笔记本内置或外接 USB）
- 网络连接（MQTT 通信需要）

## 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 首次运行会自动下载 MediaPipe 手部模型（~7.8MB）
#    存放路径: ~/.claude/models/hand_landmarker.task
```

## 运行

开**两个终端**，都 cd 到项目目录：

**终端 1** —— 启动机械狗模拟器：
```bash
python dog_simulator.py
```

**终端 2** —— 启动手势识别：
```bash
python main.py
```

对着摄像头做手势，模拟器窗口中的机械狗会响应。

### 模拟器快捷键

| 按键        | 功能                   |
| ----------- | ---------------------- |
| `S`         | 手动触发坐下（调试用） |
| `W`         | 手动触发站立           |
| `T`         | 手动触发旋转           |
| `Q` / `ESC` | 退出                   |

---

## 文件结构

```
数字图像项目/
├── main.py                 # 手势识别主程序 + 可视化界面
├── gesture_detector.py     # 手势检测核心算法
├── config.py               # 全部配置参数（阈值、滤波、MQTT）
├── mqtt_publisher.py       # MQTT 指令发布端
├── mqtt_subscriber.py      # MQTT 订阅端（纯文本日志，可选）
├── dog_simulator.py        # 机械狗可视化模拟器（OpenCV 绘制）
├── hand_landmarker.task    # MediaPipe 手部关键点模型（自动下载）
├── requirements.txt        # Python 依赖清单
└── README.md               # 本文档
```

---

## 配置说明

所有可调参数集中在 `config.py`，关键参数：

| 参数                               | 默认值             | 说明                          |
| ---------------------------------- | ------------------ | ----------------------------- |
| `FINGER_EXTENSION_RATIO_THRESHOLD` | 0.75               | 手指伸直判定阈值              |
| `VELOCITY_THRESHOLD`               | 0.03               | 垂直挥动速度阈值（归一化/帧） |
| `DISPLACEMENT_THRESHOLD`           | 0.05               | 垂直挥动位移阈值              |
| `CIRCLE_ANGLE_THRESHOLD`           | 330                | 画圈累计角度阈值（度）        |
| `CIRCLE_RADIUS_MIN`                | 0.04               | 画圈最小半径                  |
| `TRIGGER_COOLDOWN_MS`              | 1000               | 两次触发最小间隔              |
| `MQTT_BROKER`                      | test.mosquitto.org | MQTT Broker 地址              |

---

## MQTT 通信协议

| 项目  | 值                      |
| ----- | ----------------------- |
| Topic | `dog/command`           |
| QoS   | 0（最多一次，最低延迟） |
| 格式  | JSON                    |

```json
{"action": "sit", "timestamp": 1781701550.0, "velocity": 0.0375, "displacement": 0.15}
{"action": "stand", "timestamp": 1781701560.0, "velocity": -0.0320, "displacement": -0.128}
{"action": "rotate", "timestamp": 1781701570.0, "angle": 345.2}
```

---

## 依赖

```
opencv-python >= 4.8.0
mediapipe >= 0.10.0
numpy >= 1.24.0
scipy >= 1.10.0
paho-mqtt >= 2.0.0
```

---

## 树莓派部署指南

将本项目迁移到树莓派 + 真实机械狗平台，代码几乎零改动，只需以下步骤：

### 1. 树莓派环境准备

```bash
# 安装系统依赖
sudo apt update
sudo apt install -y python3-pip python3-opencv mosquitto mosquitto-clients

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 配置摄像头

```bash
# 启用摄像头模块
sudo raspi-config  # → Interface Options → Camera → Enable

# 测试摄像头
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0])"
```

### 3. 配置 MQTT Broker

```bash
# 启动 Mosquitto（树莓派作为局域网 Broker）
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# 允许局域网连接（编辑 /etc/mosquitto/mosquitto.conf，添加）
# listener 1883 0.0.0.0
# allow_anonymous true
```

### 4. 修改配置

编辑 `config.py`：

```python
# 将公共 Broker 改为树莓派本地地址
MQTT_BROKER = "192.168.x.x"   # 树莓派局域网 IP

# 若使用树莓派摄像头而非 USB 摄像头
CAMERA_INDEX = 0
```

### 5. 部署机械狗端

将 `mqtt_subscriber.py` 部署到机械狗控制板（STM32 + ESP8266）：

- **ESP8266 固件**：刷入支持 MQTT 的 AT 固件或使用 MicroPython
- **订阅主题**：`dog/command`
- **指令解析**：参考 `mqtt_subscriber.py` 中的 JSON 格式
- **动作执行**：根据 `action` 字段调用对应的舵机控制函数

```python
# 机械狗端伪代码（STM32 / MicroPython）
if action == "sit":
    set_servo(front_legs, angle=90)   # 前腿弯曲
    set_servo(back_legs, angle=45)    # 后腿折叠
elif action == "stand":
    set_servo(all_legs, angle=0)      # 所有腿伸直
elif action == "rotate":
    rotate_in_place(360)              # 原地转身
```

### 6. 运行

```bash
# 树莓派上启动手势识别
python3 main.py

# 机械狗端启动订阅（已在 STM32/ESP8266 上运行）
```

### 端到端延迟

| 环节                    | 延迟                               |
| ----------------------- | ---------------------------------- |
| 手势检测                | ~33ms（30 FPS）                    |
| 算法判定                | ~5ms                               |
| MQTT 传输（局域网有线） | <10ms                              |
| 舵机响应                | ~50ms                              |
| **总计**                | **~100ms**（满足 <150ms 设计目标） |

---

## 项目状态

- [x] 手势识别原型（平摊向下/向上 + 握拳画圈）
- [x] MQTT 通信（发布/订阅）
- [x] 可视化模拟器（站立/坐下/转身动画）
- [ ] 树莓派实机部署
- [ ] 真实机械狗联调
