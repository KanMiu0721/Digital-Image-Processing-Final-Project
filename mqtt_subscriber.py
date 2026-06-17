"""
MQTT 订阅端 —— 模拟机械狗控制板接收指令

独立运行，订阅 dog/command 主题，
收到指令后打印并模拟执行动作。

使用方法：
    python mqtt_subscriber.py
"""

import json
import time
import paho.mqtt.client as mqtt

# ---- 配置 ----
MQTT_BROKER = "test.mosquitto.org"  # 生产环境改为机械狗局域网 IP
MQTT_PORT = 1883
MQTT_TOPIC = "dog/command"
MQTT_QOS = 0


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[订阅端] 已连接到 Broker: {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC, qos=MQTT_QOS)
        print(f"[订阅端] 已订阅主题: {MQTT_TOPIC}")
    else:
        print(f"[订阅端] 连接失败: {reason_code}")


def on_message(client, userdata, msg):
    """收到指令时执行"""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        action = payload.get("action", "unknown")
        timestamp = payload.get("timestamp", 0)
        velocity = payload.get("velocity", 0)
        displacement = payload.get("displacement", 0)

        print()
        print("=" * 50)
        print(f"  🐕 机械狗收到指令!")
        print(f"  动作:  {action}")
        print(f"  速度:  {velocity:.4f}  位移: {displacement:.4f}")
        print(f"  时间戳: {timestamp:.0f}")
        print("=" * 50)

        # ---- 模拟动作执行 ----
        if action == "sit":
            print("  [执行] 舵机调整中... 前腿弯曲 → 后腿折叠 → 坐下完成 ✓")
        elif action == "stand":
            print("  [执行] 舵机调整中... 站立完成 ✓")
        else:
            print(f"  [执行] 未知动作: {action}")

    except json.JSONDecodeError:
        print(f"[订阅端] 收到无效 JSON: {msg.payload}")
    except Exception as e:
        print(f"[订阅端] 处理消息出错: {e}")


def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"[订阅端] 已断开连接 (原因码: {reason_code})")


def main():
    print("=" * 50)
    print("  机械狗 MQTT 订阅端（模拟）")
    print("  等待手势指令...")
    print("=" * 50)
    print()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="dog_subscriber",
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        print("[订阅端] 开始监听... (Ctrl+C 退出)")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[订阅端] 退出...")
    except Exception as e:
        print(f"[订阅端] 错误: {e}")
    finally:
        client.disconnect()
        print("[订阅端] 已关闭")


if __name__ == "__main__":
    main()
