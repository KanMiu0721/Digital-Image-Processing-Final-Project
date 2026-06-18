"""
手势检测核心模块

支持三种手势：
1. 平摊手掌向下挥 → SIT（坐下）
2. 平摊手掌向上挥 → STAND（站立）
3. 握拳画圈 → ROTATE（旋转）

检测方法：
- 手掌形状：距离比值法（math.hypot 零分配）
- 垂直运动：滑动窗口 + Savitzky-Golay 滤波
- 圆形轨迹：累计转角法（滑动窗口内角度积分）
"""

import math
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import time

# ---- scipy 安全导入 ----
try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import config


@dataclass
class GestureResult:
    """手势检测结果"""
    command: Optional[str]          # 'SIT' / 'STAND' / 'ROTATE' / None
    is_palm_flat: bool
    is_fist: bool                   # 是否握拳
    is_moving_down: bool
    is_moving_up: bool
    is_circling: bool               # 是否在画圈
    velocity: float
    displacement: float
    circle_angle: float             # 画圈累计角度（度）
    extended_fingers: int
    timestamp: float


class GestureDetector:
    """多手势检测器"""

    FINGER_INDICES = [
        (5, 6, 7, 8),      # 食指
        (9, 10, 11, 12),   # 中指
        (13, 14, 15, 16),  # 无名指
        (17, 18, 19, 20),  # 小指
    ]

    def __init__(self):
        self.wrist_y_history: deque = deque(maxlen=config.SLIDING_WINDOW_SIZE)
        self.wrist_xy_history: deque = deque(maxlen=config.CIRCLE_WINDOW_SIZE)
        self.wrist_trail: deque = deque(maxlen=config.TRAIL_LENGTH)
        self.last_trigger_time: float = 0.0
        self._prev_extended: int = -1     # 上一帧伸直手指数
        self._suppress_down_until: float = 0.0   # 禁止 SIT 直到（monotonic ms）
        self._suppress_up_until: float = 0.0     # 禁止 STAND 直到
        self._fist_frame_count: int = 0          # 握拳连续帧数

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(self, landmarks, image_height: int, image_width: int) -> Optional[GestureResult]:
        if landmarks is None:
            self.wrist_y_history.clear()
            self.wrist_xy_history.clear()
            return None

        # ---- 1. 手型判定 ----
        is_palm_flat, extended_fingers = self._check_palm_flat(landmarks)
        is_fist = (extended_fingers <= config.FIST_MAX_EXTENDED_FINGERS)

        # 手型切换时清空画圈历史（防止挥手轨迹误触发旋转）
        shape_changed = (self._prev_extended >= 0 and
                         ((self._prev_extended >= 3 and extended_fingers <= 0) or
                          (self._prev_extended <= 0 and extended_fingers >= 3)))
        if shape_changed:
            self.wrist_xy_history.clear()

        # ---- 2. 手腕轨迹 ----
        wrist = landmarks[0]
        self.wrist_y_history.append(wrist.y)
        self.wrist_xy_history.append((wrist.x, wrist.y))
        self.wrist_trail.append((wrist.x * image_width, wrist.y * image_height))

        # ---- 3. 运动判定 ----
        is_moving_down, is_moving_up, velocity, displacement = self._check_vertical_motion()
        is_circling, circle_angle = self._check_circular_motion()

        self._prev_extended = extended_fingers

        # ---- 4. 握拳帧计数（需连续N帧才允许画圈）----
        if is_fist:
            self._fist_frame_count += 1
        else:
            self._fist_frame_count = 0
        fist_held = self._fist_frame_count >= config.FIST_HOLD_FRAMES

        # ---- 5. 指令仲裁（优先级: ROTATE > STAND > SIT）----
        command = None
        current_time = time.monotonic() * 1000.0
        cooldown_ok = (current_time - self.last_trigger_time > config.TRIGGER_COOLDOWN_MS)

        if cooldown_ok:
            # ROTATE: 握拳+持续+画圈
            if is_fist and fist_held and is_circling:
                command = 'ROTATE'
            elif is_palm_flat:
                # 静止起始判定：窗口最旧2帧Y值靠近 → 运动从静止开始，不是乱晃
                y_list = list(self.wrist_y_history)
                still_start = (max(y_list[:2]) - min(y_list[:2])) < config.STILL_DISPERSION_THRESHOLD
                # SIT: 向下 + 静止起始 + 不在反弹抑制期内
                if is_moving_down and still_start and current_time > self._suppress_down_until:
                    command = 'SIT'
                # STAND: 向上 + 静止起始 + 不在反弹抑制期内
                elif is_moving_up and still_start and current_time > self._suppress_up_until:
                    command = 'STAND'

            if command is not None:
                self.last_trigger_time = current_time
                # 短时间抑制反方向（防手自然收回误触发），但不清理历史
                if command == 'SIT':
                    self._suppress_up_until = current_time + config.DIRECTION_SUPPRESS_MS
                elif command == 'STAND':
                    self._suppress_down_until = current_time + config.DIRECTION_SUPPRESS_MS
                elif command == 'ROTATE':
                    self.wrist_xy_history.clear()

        return GestureResult(
            command=command,
            is_palm_flat=is_palm_flat,
            is_fist=is_fist,
            is_moving_down=is_moving_down,
            is_moving_up=is_moving_up,
            is_circling=is_circling,
            velocity=velocity,
            displacement=displacement,
            circle_angle=circle_angle,
            extended_fingers=extended_fingers,
            timestamp=current_time,
        )

    # ------------------------------------------------------------------
    # 手型判定
    # ------------------------------------------------------------------

    def _check_palm_flat(self, landmarks) -> Tuple[bool, int]:
        extended_count = 0
        for mcp_idx, pip_idx, dip_idx, tip_idx in self.FINGER_INDICES:
            if self._is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx):
                extended_count += 1
        return extended_count >= config.MIN_EXTENDED_FINGERS, extended_count

    @staticmethod
    def _is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        def _dist(a, b):
            return math.hypot(b.x - a.x, b.y - a.y)
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        dip = landmarks[dip_idx]
        tip = landmarks[tip_idx]
        path_length = _dist(mcp, pip) + _dist(pip, dip) + _dist(dip, tip)
        direct_length = _dist(mcp, tip)
        if path_length < 1e-6:
            return False
        return direct_length / path_length > config.FINGER_EXTENSION_RATIO_THRESHOLD

    # ------------------------------------------------------------------
    # 垂直运动判定（向上 + 向下）
    # ------------------------------------------------------------------

    def _check_vertical_motion(self) -> Tuple[bool, bool, float, float]:
        """
        Returns: (is_moving_down, is_moving_up, velocity, displacement)
        velocity > 0 = 向下, velocity < 0 = 向上
        """
        if len(self.wrist_y_history) < config.SLIDING_WINDOW_SIZE:
            return False, False, 0.0, 0.0

        y_values = np.array(self.wrist_y_history, dtype=np.float64)
        smoothed = self._apply_filter(y_values)

        displacement = float(smoothed[-1] - smoothed[0])
        n_intervals = len(smoothed) - 1
        velocity = displacement / n_intervals if n_intervals > 0 else 0.0

        is_moving_down = (velocity > config.VELOCITY_THRESHOLD and
                          displacement > config.DISPLACEMENT_THRESHOLD)
        # 向上挥使用更低阈值（人向上挥天然比向下慢）
        up_v = config.VELOCITY_THRESHOLD * config.UPWARD_THRESHOLD_RATIO
        up_d = config.DISPLACEMENT_THRESHOLD * config.UPWARD_THRESHOLD_RATIO
        is_moving_up = (velocity < -up_v and displacement < -up_d)
        return is_moving_down, is_moving_up, velocity, displacement

    # ------------------------------------------------------------------
    # 画圈判定（累计转角法）
    # ------------------------------------------------------------------

    def _check_circular_motion(self) -> Tuple[bool, float]:
        """
        累计转角法：计算手腕轨迹在滑动窗口内的累计转向角度。
        角度 > CIRCLE_ANGLE_THRESHOLD（默认300°）且轨迹半径够大 → 画圈。

        Returns: (is_circling, total_angle_degrees)
        """
        if len(self.wrist_xy_history) < config.CIRCLE_WINDOW_SIZE:
            return False, 0.0

        pts = list(self.wrist_xy_history)

        # 检查轨迹半径（过滤手抖微动）
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        max_radius = max(math.hypot(x - cx, y - cy) for x, y in pts)
        if max_radius < config.CIRCLE_RADIUS_MIN:
            return False, 0.0

        # 累计转角
        total_angle = 0.0
        for i in range(len(pts) - 2):
            a, b, c = pts[i], pts[i + 1], pts[i + 2]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            total_angle += math.atan2(cross, dot)

        total_deg = abs(math.degrees(total_angle))
        is_circling = total_deg > config.CIRCLE_ANGLE_THRESHOLD
        return is_circling, total_deg

    # ------------------------------------------------------------------
    # 滤波
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_filter(y_values: np.ndarray) -> np.ndarray:
        window_len = min(config.SG_WINDOW_LENGTH, len(y_values))
        if window_len % 2 == 0:
            window_len -= 1
        polyorder = config.SG_POLYORDER
        if polyorder >= window_len:
            polyorder = window_len - 1
        if window_len < 3 or not _HAS_SCIPY:
            kernel = np.ones(window_len) / window_len
            return np.convolve(y_values, kernel, mode='same')
        return savgol_filter(y_values, window_len, polyorder)

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self):
        self.wrist_y_history.clear()
        self.wrist_xy_history.clear()
        self.wrist_trail.clear()
        self.last_trigger_time = 0.0
        self._prev_extended = -1
        self._suppress_down_until = 0.0
        self._suppress_up_until = 0.0
        self._fist_frame_count = 0
