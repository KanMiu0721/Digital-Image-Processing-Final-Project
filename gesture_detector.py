"""
手势检测核心模块

实现两大核心功能：
1. 手掌平摊静态判定 —— 基于手指伸直程度（距离比值法）
2. 手掌向下挥动轨迹分析 —— 基于手腕关键点的滑动窗口 + Savitzky-Golay滤波
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

from scipy.signal import savgol_filter

import config


@dataclass
class GestureResult:
    """手势检测结果"""
    command: Optional[str]          # 触发指令 ('SIT') 或 None
    is_palm_flat: bool              # 当前帧是否手掌平摊
    is_moving_down: bool            # 当前窗口是否向下挥动
    velocity: float                 # 当前速度（归一化坐标/帧）
    displacement: float             # 窗口内总位移
    extended_fingers: int           # 伸直的手指数量
    timestamp: float                # 时间戳


class GestureDetector:
    """
    手势检测器

    采用状态机模式跟踪手势序列：
    - 持续检测手掌平摊状态
    - 在滑动窗口内分析手腕垂直运动
    - 两个条件同时满足时触发指令

    使用方法:
        detector = GestureDetector()
        result = detector.detect(landmarks, height, width)
        if result.command == 'SIT':
            print("机械狗坐下!")
    """

    # MediaPipe 手指关键点索引: (MCP, PIP, DIP, TIP)
    FINGER_INDICES = [
        (5, 6, 7, 8),      # 食指 Index
        (9, 10, 11, 12),   # 中指 Middle
        (13, 14, 15, 16),  # 无名指 Ring
        (17, 18, 19, 20),  # 小指 Pinky
    ]

    FINGER_NAMES = ["食指", "中指", "无名指", "小指"]

    def __init__(self):
        # 手腕Y坐标历史（归一化值），用于滑动窗口分析
        self.wrist_y_history: deque = deque(maxlen=config.SLIDING_WINDOW_SIZE)
        # 手腕轨迹历史（像素坐标），用于可视化
        self.wrist_trail: deque = deque(maxlen=config.TRAIL_LENGTH)
        # 上次触发时间（毫秒）
        self.last_trigger_time: float = 0
        # 当前帧状态
        self.is_palm_flat: bool = False
        self.is_moving_down: bool = False
        self.extended_fingers: int = 0
        # 采样计数（用于稳定滤波）
        self.frame_count: int = 0

    def detect(
        self,
        landmarks,
        image_height: int,
        image_width: int
    ) -> Optional[GestureResult]:
        """
        主检测入口：处理一帧手部关键点数据

        Args:
            landmarks: MediaPipe NormalizedLandmark 列表（21个点）
            image_height: 图像高度（像素）
            image_width: 图像宽度（像素）

        Returns:
            GestureResult 或 None（无手部检测时）
        """
        if landmarks is None:
            return None

        self.frame_count += 1

        # ---- 1. 手掌平摊判定 ----
        self.is_palm_flat, self.extended_fingers = self._check_palm_flat(landmarks)

        # ---- 2. 手腕轨迹记录 ----
        wrist = landmarks[0]
        wrist_y_norm = wrist.y  # MediaPipe已归一化到[0,1]
        self.wrist_y_history.append(wrist_y_norm)

        # 保存像素坐标用于可视化轨迹
        wrist_px = (wrist.x * image_width, wrist.y * image_height)
        self.wrist_trail.append(wrist_px)

        # ---- 3. 向下挥动判定 ----
        self.is_moving_down, velocity, displacement = self._check_downward_motion()

        # ---- 4. 指令触发判定 ----
        command = None
        current_time = time.time() * 1000  # 毫秒

        if self.is_palm_flat and self.is_moving_down:
            if current_time - self.last_trigger_time > config.TRIGGER_COOLDOWN_MS:
                command = 'SIT'
                self.last_trigger_time = current_time

        return GestureResult(
            command=command,
            is_palm_flat=self.is_palm_flat,
            is_moving_down=self.is_moving_down,
            velocity=velocity,
            displacement=displacement,
            extended_fingers=self.extended_fingers,
            timestamp=current_time,
        )

    def _check_palm_flat(self, landmarks) -> Tuple[bool, int]:
        """
        检查手掌是否平摊（手指伸直）

        方法：距离比值法
        - 计算每根手指的指节路径总长度 与 指尖→指根直线距离 的比值
        - 伸直手指：比值接近 1.0（直线距离 ≈ 路径长度）
        - 弯曲手指：比值接近 0.0（直线距离 << 路径长度）

        Returns:
            (is_flat, extended_count): 是否平摊, 伸直手指数量
        """
        extended_count = 0

        for mcp_idx, pip_idx, dip_idx, tip_idx in self.FINGER_INDICES:
            if self._is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx):
                extended_count += 1

        is_flat = extended_count >= config.MIN_EXTENDED_FINGERS
        return is_flat, extended_count

    def _is_finger_extended(
        self,
        landmarks,
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int
    ) -> bool:
        """
        判断单根手指是否伸直（距离比值法）

        几何原理：
        - 伸直时 3段指节趋近共线：|MCP→TIP| ≈ |MCP→PIP| + |PIP→DIP| + |DIP→TIP|
        - 弯曲时指尖回折：|MCP→TIP| << 路径总长
        - ratio = 直线距离 / 路径总长，伸直时 → 1.0，弯曲时 → <0.5
        """
        mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y])
        pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y])
        dip = np.array([landmarks[dip_idx].x, landmarks[dip_idx].y])
        tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y])

        # 三段指节路径总长
        path_length = (
            np.linalg.norm(pip - mcp) +
            np.linalg.norm(dip - pip) +
            np.linalg.norm(tip - dip)
        )

        # 指尖到指根直线距离
        direct_length = np.linalg.norm(tip - mcp)

        if path_length < 1e-6:
            return False

        ratio = direct_length / path_length
        return ratio > config.FINGER_EXTENSION_RATIO_THRESHOLD

    def _check_downward_motion(self) -> Tuple[bool, float, float]:
        """
        检查手掌是否向下挥动

        方法：滑动窗口 + Savitzky-Golay 滤波
        - 维护最近N帧的手腕Y坐标
        - 对窗口数据做SG滤波平滑
        - 计算滤波后的速度和总位移
        - 速度 > 阈值 且 位移 > 阈值 → 判定为向下挥动

        Returns:
            (is_moving_down, velocity, displacement)
        """
        if len(self.wrist_y_history) < config.SLIDING_WINDOW_SIZE:
            return False, 0.0, 0.0

        y_values = np.array(self.wrist_y_history, dtype=np.float64)

        # Savitzky-Golay 滤波 —— 抑制高频抖动，保留运动趋势
        # 确保窗口长度不超过数据点数，且为奇数
        window_len = min(config.SG_WINDOW_LENGTH, len(y_values))
        if window_len % 2 == 0:
            window_len -= 1
        if window_len < 3:
            # 数据不够，直接用原始值
            smoothed = y_values
        else:
            smoothed = savgol_filter(y_values, window_len, config.SG_POLYORDER)

        # 计算总位移（正=向下，Y轴在图像中朝下）
        displacement = float(smoothed[-1] - smoothed[0])

        # 平均每帧速度
        velocity = displacement / len(smoothed)

        is_moving_down = (
            velocity > config.VELOCITY_THRESHOLD and
            displacement > config.DISPLACEMENT_THRESHOLD
        )

        return is_moving_down, velocity, displacement

    def reset(self):
        """重置检测器状态"""
        self.wrist_y_history.clear()
        self.wrist_trail.clear()
        self.last_trigger_time = 0
        self.is_palm_flat = False
        self.is_moving_down = False
        self.extended_fingers = 0
        self.frame_count = 0
