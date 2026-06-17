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
import numpy as np
import time
import sys

from gesture_detector import GestureDetector, GestureResult
import config


# ============================================================
# MediaPipe 手部连接关系（用于绘制骨架）
# ============================================================
HAND_CONNECTIONS = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 手掌横向连接
    (5, 9), (9, 13), (13, 17),
]


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
            landmarks: MediaPipe手部关键点（归一化坐标）
            result: 当前帧手势检测结果
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

        # ---- 绘制状态面板（返回新图像，因为叠加操作创建新数组）----
        image = self._draw_status_panel(image, state_text, state_color, fps, result, detector)

        # ---- 触发闪烁效果 ----
        if result is not None and result.command == 'SIT':
            self._draw_trigger_flash(image)

        return image

    def _draw_trail(self, image, trail):
        """绘制手腕运动轨迹（带渐隐效果）"""
        pts = list(trail)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            alpha = i / n  # 越近越亮
            color = tuple(int(c * alpha) for c in config.COLOR_TRAIL)
            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            pt2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

    def _draw_hand_skeleton(self, image, landmarks, h, w, color):
        """绘制手部骨架连线"""
        # 先画连线
        for start_idx, end_idx in HAND_CONNECTIONS:
            x1 = int(landmarks[start_idx].x * w)
            y1 = int(landmarks[start_idx].y * h)
            x2 = int(landmarks[end_idx].x * w)
            y2 = int(landmarks[end_idx].y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 再画关键点
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(image, (cx, cy), config.LANDMARK_CIRCLE_RADIUS, color, -1, cv2.LINE_AA)

        # 手腕用更大的圆突出显示
        wrist_x = int(landmarks[0].x * w)
        wrist_y = int(landmarks[0].y * h)
        cv2.circle(image, (wrist_x, wrist_y), config.LANDMARK_CIRCLE_RADIUS + 3,
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

    def _draw_status_panel(self, image, state_text, state_color, fps, result, detector):
        """绘制状态信息面板（返回叠加后的图像）"""
        h, w = image.shape[:2]

        # 半透明背景面板
        overlay = image.copy()
        panel_h = 120 if self.debug_mode else 90
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

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

    def _draw_trigger_flash(self, image):
        """触发指令时的红色闪烁边框"""
        h, w = image.shape[:2]
        thickness = 8
        cv2.rectangle(image, (0, 0), (w, h), config.COLOR_TRIGGERED, thickness)
        # 中央大字提示
        text = "SIT! / 坐下!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(image, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_TRIGGERED, 3)


def init_camera():
    """初始化摄像头"""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (index={config.CAMERA_INDEX})")
        print("  请检查摄像头是否被其他程序占用，或尝试修改 config.py 中的 CAMERA_INDEX")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 摄像头已打开: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f}fps")

    return cap


def main():
    print("=" * 60)
    print("  机械狗手势控制系统 —— 手势识别原型")
    print("  手势: 平摊手掌向下挥 → 机械狗坐下")
    print("=" * 60)
    print()
    print("[INFO] 初始化 MediaPipe Hands...")

    # ---- 初始化 ----
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=config.STATIC_IMAGE_MODE,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        model_complexity=config.MODEL_COMPLEXITY,
    )

    detector = GestureDetector()
    visualizer = Visualizer()

    cap = init_camera()

    print("[INFO] 初始化完成，开始手势检测...")
    print("[INFO] 请对着摄像头做「平摊手掌向下挥」手势")
    print()

    # ---- FPS 计时 ----
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    try:
        while True:
            # ---- 读取帧 ----
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 读取帧失败，跳过...")
                continue

            # 镜像翻转（更自然）
            frame = cv2.flip(frame, 1)

            # BGR → RGB（MediaPipe 需要 RGB 输入）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # 性能优化：标记为不可写

            # ---- MediaPipe 手部检测 ----
            mp_result = hands.process(rgb_frame)

            rgb_frame.flags.writeable = True

            # ---- 手势检测 ----
            if mp_result.multi_hand_landmarks:
                # 只取第一只手（config.MAX_NUM_HANDS=1）
                hand_landmarks = mp_result.multi_hand_landmarks[0]
                result = detector.detect(
                    hand_landmarks.landmark,
                    config.CAMERA_HEIGHT,
                    config.CAMERA_WIDTH,
                )

                # ---- 指令输出 ----
                if result is not None and result.command == 'SIT':
                    print(f"  🐕 机械狗坐下!  "
                          f"(速度={result.velocity:.4f}, 位移={result.displacement:.4f})")
            else:
                hand_landmarks = None
                result = None

            # ---- 可视化 ----
            display_frame = visualizer.draw(
                frame,
                hand_landmarks.landmark if hand_landmarks else None,
                result,
                detector,
                current_fps,
            )

            # ---- FPS 计算 ----
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

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
        hands.close()
        print("[INFO] 资源已释放，程序结束")


if __name__ == "__main__":
    main()
