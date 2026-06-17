"""
手势识别原型 —— 主程序

功能：
- 调用笔记本摄像头捕获实时视频
- 使用 MediaPipe Hands 提取21个手部关键点
- 实时检测"平摊手掌向下挥"手势
- 可视化显示：关键点、手腕轨迹、手势状态、FPS
- 检测到手势时在终端输出 "🐕 机械狗坐下!"

按键控制：
  q / ESC  →  退出程序
  r        →  重置检测器
  d        →  切换调试模式（显示更多信息）

使用方法：
  python main.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions
import numpy as np
import time
import sys
import os

from gesture_detector import GestureDetector, GestureResult
from mqtt_publisher import MqttPublisher
import config

# 模型文件路径 —— 优先放在不含中文的目录（MediaPipe C++ 层编码限制）
#   本地路径作为备用（Linux/Mac 上无此问题）
_MODEL_PATH_LOCAL = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
_MODEL_PATH_SAFE = os.path.join(os.path.expanduser('~'), '.claude', 'models', 'hand_landmarker.task')
_MODEL_PATH = _MODEL_PATH_SAFE if os.path.exists(_MODEL_PATH_SAFE) else _MODEL_PATH_LOCAL


class Visualizer:
    """可视化渲染器 —— 在视频帧上叠加手势信息"""

    def __init__(self):
        self.debug_mode = False

    def draw(
        self,
        image: np.ndarray,
        landmarks,
        result: GestureResult,
        detector: GestureDetector,
        fps: float,
    ) -> np.ndarray:
        """
        在图像上绘制所有可视化元素

        Args:
            image: 原始BGR图像
            landmarks: MediaPipe手部关键点（归一化坐标），可为 None
            result: 当前帧手势检测结果，可为 None
            detector: 手势检测器（用于获取轨迹历史）
            fps: 当前帧率

        Returns:
            叠加了可视化元素的图像
        """
        h, w = image.shape[:2]

        # ---- 选择当前状态颜色 ----
        if result is not None and result.command == 'SIT':
            state_color = config.COLOR_TRIGGERED
            state_text = "SIT! (坐下!)"
        elif result is not None and result.is_moving_down:
            state_color = config.COLOR_MOVING
            state_text = "DOWN (向下)"
        elif result is not None and result.is_palm_flat:
            state_color = config.COLOR_PALM_FLAT
            state_text = "PALM FLAT (平摊)"
        else:
            state_color = config.COLOR_IDLE
            state_text = "IDLE (待机)"

        # ---- 绘制手腕轨迹 ----
        if result is not None:
            self._draw_trail(image, detector.wrist_trail)

        # ---- 绘制手部骨架和关键点 ----
        if landmarks is not None:
            self._draw_hand_skeleton(image, landmarks, h, w, state_color)
            self._draw_landmark_labels(image, landmarks, h, w)

        # ---- 绘制状态面板 ----
        image = self._draw_status_panel(image, state_text, state_color, fps, result)

        # ---- 触发闪烁效果 ----
        if result is not None and result.command == 'SIT':
            self._draw_trigger_flash(image)

        return image

    # ------------------------------------------------------------------
    # 轨迹绘制
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_trail(image, trail):
        """绘制手腕运动轨迹（带渐隐效果，最新点最亮）"""
        pts = list(trail)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            # alpha: 最新线段 = 1.0, 最旧线段 ≈ 1/(n-1)
            alpha = i / (n - 1)
            color = tuple(int(c * alpha) for c in config.COLOR_TRAIL)
            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            pt2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 手部骨架绘制
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_hand_skeleton(image, landmarks, h, w, color):
        """绘制手部骨架连线（预计算像素坐标，避免重复计算）"""
        # 一次性预计算所有 21 个关键点的像素坐标
        px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # 画连线（Tasks API 的 Connection 使用 .start / .end）
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, px[conn.start], px[conn.end], color, 2, cv2.LINE_AA)

        # 画关键点
        for cx, cy in px:
            cv2.circle(image, (cx, cy), config.LANDMARK_CIRCLE_RADIUS,
                       color, -1, cv2.LINE_AA)

        # 手腕用更大的圆突出显示
        wx, wy = px[0]
        cv2.circle(image, (wx, wy), config.LANDMARK_CIRCLE_RADIUS + 3,
                   color, 2, cv2.LINE_AA)

    def _draw_landmark_labels(self, image, landmarks, h, w):
        """调试模式：显示关键点编号"""
        if not self.debug_mode:
            return
        for i, lm in enumerate(landmarks):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.putText(image, str(i), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # 状态面板
    # ------------------------------------------------------------------

    def _draw_status_panel(self, image, state_text, state_color, fps, result):
        """绘制状态信息面板（ROI 局部叠加，避免全帧拷贝）"""
        h, w = image.shape[:2]
        panel_h = 120 if self.debug_mode else 90

        # 仅对面板区域做半透明叠加（而非全帧 copy + addWeighted）
        roi = image[0:panel_h, 0:w]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, 0.5, roi, 0.5, 0)
        image[0:panel_h, 0:w] = blended

        y_offset = 25

        # FPS
        cv2.putText(image, f"FPS: {fps:.0f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE, (0, 255, 255), config.FONT_THICKNESS)

        # 状态
        y_offset += 25
        cv2.putText(image, f"State: {state_text}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE, state_color, config.FONT_THICKNESS)

        if result is not None:
            y_offset += 20
            cv2.putText(image,
                        f"Fingers: {result.extended_fingers}/4 | "
                        f"Vel: {result.velocity:.4f} | "
                        f"Disp: {result.displacement:.4f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, config.COLOR_TEXT, 1)

            if self.debug_mode:
                y_offset += 18
                cv2.putText(image,
                            f"Thresholds: vel>{config.VELOCITY_THRESHOLD:.3f} "
                            f"disp>{config.DISPLACEMENT_THRESHOLD:.3f}",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (180, 180, 180), 1)

        # 操作提示
        y_offset = h - 10
        cv2.putText(image, "q/ESC:Quit | r:Reset | d:Debug",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (150, 150, 150), 1)

        return image

    # ------------------------------------------------------------------
    # 触发闪烁
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_trigger_flash(image):
        """触发指令时的红色闪烁边框 + 中央大字"""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w, h), config.COLOR_TRIGGERED, 8)
        text = "SIT! / 坐下!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(image, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_TRIGGERED, 3)


# ======================================================================
# 摄像头初始化
# ======================================================================

def download_model():
    """下载 MediaPipe 手部关键点模型（.task 文件）"""
    if os.path.exists(_MODEL_PATH):
        return

    import urllib.request
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task')
    print(f"[INFO] 正在下载手部关键点模型...")
    print(f"  {url}")
    try:
        urllib.request.urlretrieve(url, _MODEL_PATH)
        print(f"[INFO] 模型已下载: {_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] 模型下载失败: {e}")
        print("  请手动下载并放置到:", _MODEL_PATH)
        sys.exit(1)


def init_camera():
    """初始化摄像头，返回 (cap, actual_w, actual_h)"""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (index={config.CAMERA_INDEX})")
        print("  请检查摄像头是否被其他程序占用，或尝试修改 config.py 中的 CAMERA_INDEX")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    return cap, actual_w, actual_h


# ======================================================================
# 主循环
# ======================================================================

def main():
    print("=" * 60)
    print("  机械狗手势控制系统 —— 手势识别原型")
    print("  手势: 平摊手掌向下挥 → 机械狗坐下")
    print("=" * 60)
    print()

    # ---- 下载模型 ----
    download_model()

    print("[INFO] 初始化 MediaPipe HandLandmarker...")

    # ---- 初始化 MediaPipe HandLandmarker（Tasks API）----
    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        hand_landmarker = HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[ERROR] MediaPipe HandLandmarker 初始化失败: {e}")
        print("  请检查模型文件是否存在:", _MODEL_PATH)
        sys.exit(1)

    detector = GestureDetector()
    visualizer = Visualizer()
    mqtt = MqttPublisher()
    mqtt.connect()  # 非阻塞，后台连接
    cap, cam_w, cam_h = init_camera()

    print("[INFO] 初始化完成，开始手势检测...")
    print("[INFO] 请对着摄像头做「平摊手掌向下挥」手势")
    print()

    # ---- FPS 计时（使用 monotonic 免疫时钟跳变）----
    fps_start_time = time.monotonic()
    fps_frame_count = 0
    current_fps = 0.0

    try:
        while True:
            # ---- 读取帧 ----
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 读取帧失败，跳过...")
                cv2.waitKey(1)
                time.sleep(0.01)
                continue

            # 镜像翻转（更自然）
            frame = cv2.flip(frame, 1)

            # 使用实际帧尺寸（而非 config 的硬编码值）
            actual_h, actual_w = frame.shape[:2]

            # BGR → RGB，创建 MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # ---- MediaPipe 手部检测（Tasks API）----
            mp_result = hand_landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000))

            # ---- 手势检测（无论有无手部都调用 detect）----
            if mp_result.hand_landmarks:
                hand_landmarks = mp_result.hand_landmarks[0]
                result = detector.detect(
                    hand_landmarks,
                    actual_h,
                    actual_w,
                )
                if result is not None and result.command:
                    cmd = result.command
                    if cmd == 'SIT':
                        print(f"  🐕 坐下!  "
                              f"(v={result.velocity:.4f}, d={result.displacement:.4f})")
                        mqtt.send_sit(velocity=result.velocity,
                                      displacement=result.displacement)
                    elif cmd == 'STAND':
                        print(f"  🐕 站立!  "
                              f"(v={result.velocity:.4f}, d={result.displacement:.4f})")
                        mqtt.send_stand(velocity=result.velocity,
                                        displacement=result.displacement)
                    elif cmd == 'ROTATE':
                        print(f"  🐕 旋转!  "
                              f"(角度={result.circle_angle:.0f}°)")
                        mqtt.send_rotate(angle=result.circle_angle)
            else:
                hand_landmarks = None
                # 传入 None 清空滑动窗口，防止手部再现时误触发
                result = detector.detect(None, actual_h, actual_w)

            # ---- 可视化 ----
            display_frame = visualizer.draw(
                frame,
                hand_landmarks,
                result,
                detector,
                current_fps,
            )

            # ---- FPS 计算 ----
            fps_frame_count += 1
            elapsed = time.monotonic() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.monotonic()

            # ---- 显示 ----
            cv2.imshow("Gesture Control - Sit Detection", display_frame)

            # ---- 按键处理 ----
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC
                print("[INFO] 用户退出")
                break
            elif key == ord('r'):
                detector.reset()
                print("[INFO] 检测器已重置")
            elif key == ord('d'):
                visualizer.debug_mode = not visualizer.debug_mode
                print(f"[INFO] 调试模式: {'ON' if visualizer.debug_mode else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[INFO] 检测到 Ctrl+C，退出...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        mqtt.disconnect()
        print("[INFO] 资源已释放，程序结束")


if __name__ == "__main__":
    main()
