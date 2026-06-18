"""
机械狗可视化模拟器 —— 第三步

用 OpenCV 绘制简笔画机械狗，通过 MQTT 接收手势指令，
实现站立 → 坐下 的平滑动画。

使用方法：
    python dog_simulator.py
"""

import cv2
import numpy as np
import json
import time
import paho.mqtt.client as mqtt
from collections import deque

# ---- 配置 ----
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "dog/command"
WINDOW_NAME = "Robot Dog Simulator"
WINDOW_W, WINDOW_H = 600, 400

# 颜色 (BGR)
BG_COLOR = (40, 40, 50)
DOG_COLOR = (220, 200, 100)
DOG_COLOR_ACTIVE = (100, 255, 100)
TEXT_COLOR = (220, 220, 220)
FLOOR_COLOR = (80, 80, 100)


class RobotDog:
    """
    机械狗模型 —— 显式定义站立/坐下两套骨骼坐标，干净插值。

    站立:                      坐下:
      [头]                       [头]
      [身]                       [身]
      |   |                     /    \
      |   |                    /      \
    前腿  后腿              前腿(弯)  后腿(折)
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.floor_y = int(h * 0.78)

        # ---- 站立骨骼 ----
        cx, cy = int(w * 0.42), self.floor_y - 110
        self.STAND = {
            # 躯干四点 (TL, TR, BR, BL) —— 平坦矩形
            'body':    ((cx, cy), (cx + 100, cy),
                        (cx + 100, cy + 28), (cx, cy + 28)),
            'head':    (cx + 115, cy),
            'neck':    (cx + 100, cy),
            'f_hip':   (cx + 80, cy + 28),
            'f_knee':  (cx + 80, cy + 60),
            'f_foot':  (cx + 80, self.floor_y),
            'b_hip':   (cx + 15, cy + 28),
            'b_knee':  (cx + 15, cy + 60),
            'b_foot':  (cx + 15, self.floor_y),
            'tail':    (cx - 15, cy + 8),
            'tail_tip': (cx - 35, cy - 20),
        }

        # ---- 坐下骨骼 ----
        # 身体前高后低、细长、屁股坐地；后腿折叠贴地前伸
        sx = int(w * 0.44)
        fy = self.floor_y  # 312
        self.SIT = {
            # 躯干四点 —— 细身倾斜，BL 坐在地上
            'body':    ((sx, fy - 30), (sx + 82, fy - 60),       # TL(后顶) TR(前顶高)
                        (sx + 78, fy - 32), (sx + 4, fy)),       # BR(前底) BL(后底=地面)
            'head':    (sx + 95, fy - 78),                    # 头贴近身体，随身体下沉
            'neck':    (sx + 80, fy - 58),
            'f_hip':   (sx + 78, fy - 32),
            'f_knee':  (sx + 78, fy - 16),                       # 前腿直伸
            'f_foot':  (sx + 78, fy),                             # 前脚踩地
            'b_hip':   (sx + 5, fy - 2),                          # 后髋近地面
            'b_knee':  (sx + 26, fy),                             # 后膝贴地
            'b_foot':  (sx + 52, fy),                             # 小腿贴地前伸
            'tail':    (sx - 10, fy - 24),
            'tail_tip': (sx - 30, fy - 18),
        }

        # 动画
        self.t = 0.0          # 姿态插值
        self.target = 0.0
        self.speed = 0.10

        # 旋转动画
        self.rot_angle = 0.0  # 当前旋转角度（度）
        self.rot_target = 0.0 # 目标角度
        self.rot_speed = 12.0 # 度/帧

        # 日志
        self.last_command = None
        self.command_time = 0
        self.log = deque(maxlen=5)

    def sit(self):
        self.target = 1.0
        self.last_command = "SIT"
        self.command_time = time.monotonic()
        self.log.append(("SIT (坐下)", time.monotonic()))

    def stand(self):
        self.target = 0.0
        self.last_command = "STAND"
        self.command_time = time.monotonic()
        self.log.append(("STAND (站立)", time.monotonic()))

    def rotate(self):
        """触发原地旋转一圈"""
        self.rot_target += 360.0  # 每触发一次转 360°
        self.last_command = "ROTATE"
        self.command_time = time.monotonic()
        self.log.append(("ROTATE (旋转)", time.monotonic()))

    def update(self):
        # 姿态动画
        d = self.target - self.t
        if abs(d) < 0.003:
            self.t = self.target
        else:
            self.t += d * self.speed
        # 旋转动画
        if abs(self.rot_angle - self.rot_target) > 0.5:
            self.rot_angle += self.rot_speed * (1 if self.rot_target > self.rot_angle else -1)
            if abs(self.rot_angle - self.rot_target) < 1.0:
                self.rot_angle = self.rot_target

    def _lerp_pt(self, a, b):
        """插值两个点"""
        return (int(a[0] + (b[0] - a[0]) * self.t),
                int(a[1] + (b[1] - a[1]) * self.t))

    def _lerp_quad(self, a, b):
        """插值四边形（四个点）"""
        return tuple(
            (int(a[i][0] + (b[i][0] - a[i][0]) * self.t),
             int(a[i][1] + (b[i][1] - a[i][1]) * self.t))
            for i in range(4)
        )

    def draw(self, canvas):
        # 插值当前帧所有骨骼点
        body = self._lerp_quad(self.STAND['body'], self.SIT['body'])     # 四点四边形
        head = self._lerp_pt(self.STAND['head'], self.SIT['head'])
        f_hip = self._lerp_pt(self.STAND['f_hip'], self.SIT['f_hip'])
        f_knee = self._lerp_pt(self.STAND['f_knee'], self.SIT['f_knee'])
        f_foot = self._lerp_pt(self.STAND['f_foot'], self.SIT['f_foot'])
        b_hip = self._lerp_pt(self.STAND['b_hip'], self.SIT['b_hip'])
        b_knee = self._lerp_pt(self.STAND['b_knee'], self.SIT['b_knee'])
        b_foot = self._lerp_pt(self.STAND['b_foot'], self.SIT['b_foot'])
        tail = self._lerp_pt(self.STAND['tail'], self.SIT['tail'])
        tail_tip = self._lerp_pt(self.STAND['tail_tip'], self.SIT['tail_tip'])

        # 躯干底部中心（用于阴影）
        body_bottom_cx = (body[2][0] + body[3][0]) // 2
        body_bottom_cy = (body[2][1] + body[3][1]) // 2

        # 颜色
        flash = self.last_command and (time.monotonic() - self.command_time) < 0.5
        color = DOG_COLOR_ACTIVE if flash else DOG_COLOR

        # ---- 地面 + 阴影 ----
        cv2.line(canvas, (0, self.floor_y), (self.w, self.floor_y), FLOOR_COLOR, 1)
        shadow_rx = int(30 + self.t * 25)
        cv2.ellipse(canvas, (body_bottom_cx, self.floor_y + 3),
                    (shadow_rx, 5), 0, 0, 180, (25, 25, 35), -1)

        # ---- 尾巴 ----
        cv2.line(canvas, tail, tail_tip, color, 3)

        # ---- 后腿（两段）----
        self._draw_leg(canvas, b_hip, b_knee, b_foot, color)

        # ---- 前腿（两段）----
        self._draw_leg(canvas, f_hip, f_knee, f_foot, color)

        # ---- 躯干（四边形，前高后低）----
        body_pts = np.array(body, dtype=np.int32)
        cv2.fillPoly(canvas, [body_pts], color)
        cv2.polylines(canvas, [body_pts], True, (0, 0, 0), 1)

        # ---- 脖子连线 ----
        neck = self._lerp_pt(self.STAND['neck'], self.SIT['neck'])
        cv2.line(canvas, neck, head, color, 2)

        # ---- 头 ----
        r = 18
        hx, hy = head
        cv2.circle(canvas, (hx, hy), r, color, -1)
        cv2.circle(canvas, (hx, hy), r, (0, 0, 0), 1)

        # 眼睛（保持正常位置在头部）
        ex = hx + r // 2
        ey = hy - 4
        if self.t > 0.7:
            # 眯眼
            cv2.ellipse(canvas, (ex, ey), (3, 2), 0, 0, 180, (0, 0, 0), -1)
        else:
            cv2.circle(canvas, (ex, ey), 3, (0, 0, 0), -1)
        # 鼻子
        cv2.circle(canvas, (hx + r - 2, hy + 1), 2, (0, 0, 0), -1)
        # 耳朵
        ear_pts = np.array([
            (hx - 6, hy - r),
            (hx - 2, hy - r - 13),
            (hx + 4, hy - r),
        ])
        cv2.fillPoly(canvas, [ear_pts], color)

        # ---- HUD ----
        cv2.putText(canvas, "Robot Dog", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

        # ---- 旋转动画（水平左右翻转，模拟狗狗原地转身）----
        rotating = abs(self.rot_angle - self.rot_target) > 0.5 or abs(self.rot_angle) > 0.5
        if rotating and abs(self.rot_angle) > 0.5:
            dog_cx = (body[0][0] + body[1][0]) // 2
            # 水平缩放: cos(angle)→1→-1→1，产生左右翻转效果
            import math
            scale_x = math.cos(math.radians(self.rot_angle))
            M = np.float32([[scale_x, 0, dog_cx * (1 - scale_x)],
                            [0, 1, 0]])
            canvas[:] = cv2.warpAffine(canvas, M, (self.w, self.h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=BG_COLOR)

        if self.t < 0.05:
            state = "STANDING"
        elif self.t > 0.95:
            state = "SITTING"
            cv2.putText(canvas, "SIT !", (body[1][0] + 5, body[1][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        else:
            state = f"MOVING..."
        if rotating and abs(self.rot_angle) > 10:
            state += " + TURNING"
        cv2.putText(canvas, f"State: {state}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        # 日志
        log_y = self.h - 15
        for msg, ts in reversed(list(self.log)[-4:]):
            elapsed = time.monotonic() - ts
            alpha = max(0, 1.0 - elapsed / 3.0)
            c = tuple(int(v * alpha) for v in DOG_COLOR_ACTIVE)
            cv2.putText(canvas, f"> {msg}", (20, log_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            log_y -= 18

    def _draw_leg(self, canvas, hip, knee, foot, color):
        """绘制一条两段式腿"""
        cv2.line(canvas, hip, knee, color, 6)
        cv2.line(canvas, knee, foot, color, 5)
        cv2.circle(canvas, foot, 4, color, -1)
        cv2.circle(canvas, knee, 3, (0, 0, 0), -1)


# ======================================================================
# MQTT 回调
# ======================================================================

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[模拟器] 已连接到 Broker")
        client.subscribe(MQTT_TOPIC, qos=0)
        print(f"[模拟器] 已订阅: {MQTT_TOPIC}")
    else:
        print(f"[模拟器] 连接失败: {reason_code}")


def on_message(client, userdata, msg):
    dog = userdata
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        action = payload.get("action", "")
        print(f"[模拟器] 收到指令: {action}")
        if action == "sit":
            dog.sit()
        elif action == "stand":
            dog.stand()
        elif action == "rotate":
            dog.rotate()
    except Exception as e:
        print(f"[模拟器] 消息错误: {e}")


# ======================================================================
# 主循环
# ======================================================================

def main():
    print("=" * 50)
    print("  机械狗可视化模拟器")
    print("  等待 MQTT 指令...")
    print("=" * 50)
    print()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="dog_simulator")
    client.on_connect = on_connect
    client.on_message = on_message

    mqtt_ok = False
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_ok = True
    except Exception as e:
        print(f"[模拟器] MQTT 连接失败: {e}")
        print("[模拟器] 将以键盘模式运行（S/W/T 键手动测试）")
        print()

    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    dog = RobotDog(WINDOW_W, WINDOW_H)
    client.user_data_set(dog)
    if mqtt_ok:
        client.loop_start()

    fps_timer = time.monotonic()
    frame_count = 0

    print("[模拟器] 窗口已打开 (q 退出)")
    print()

    try:
        while True:
            dog.update()
            canvas[:] = BG_COLOR
            dog.draw(canvas)

            frame_count += 1
            if time.monotonic() - fps_timer >= 1.0:
                fps = frame_count / (time.monotonic() - fps_timer)
                frame_count = 0
                fps_timer = time.monotonic()
                cv2.setWindowTitle(WINDOW_NAME, f"Robot Dog Simulator - {fps:.0f} FPS")

            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(16) & 0xFF
            if key == ord('q') or key == 27:
                print("[模拟器] 退出")
                break
            elif key == ord('s'):
                dog.sit()
            elif key == ord('w'):
                dog.stand()
            elif key == ord('t'):       # t = turn
                dog.rotate()

    except KeyboardInterrupt:
        print("\n[模拟器] Ctrl+C 退出...")
    finally:
        if mqtt_ok:
            client.loop_stop()
            client.disconnect()
        cv2.destroyAllWindows()
        print("[模拟器] 已关闭")


if __name__ == "__main__":
    main()
