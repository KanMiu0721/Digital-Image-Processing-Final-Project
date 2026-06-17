"""
MQTT 发布端 —— 将手势指令发送到机械狗控制端

使用 paho-mqtt 客户端，发布 JSON 格式指令到 dog/command 主题。
发布操作在后台线程执行，不阻塞主循环。
"""

import json
import threading
import time
import paho.mqtt.client as mqtt

import config


class MqttPublisher:
    """
    MQTT 指令发布器

    使用方法:
        pub = MqttPublisher()
        pub.connect()
        # ... 在主循环中 ...
        pub.send_sit(velocity=0.04, displacement=0.08)
        # ... 退出时 ...
        pub.disconnect()
    """

    def __init__(self):
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=config.MQTT_CLIENT_ID,
        )
        self._connected = False
        self._lock = threading.Lock()

        # 回调
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self):
        """连接到 MQTT Broker（非阻塞，在后台线程完成握手）"""
        try:
            self._client.connect_async(config.MQTT_BROKER, config.MQTT_PORT, keepalive=60)
            self._client.loop_start()  # 启动后台网络线程
            # 等待连接完成（最多 2 秒）
            for _ in range(20):
                if self._connected:
                    print(f"[MQTT] 已连接到 Broker: {config.MQTT_BROKER}")
                    return True
                time.sleep(0.1)
            print(f"[MQTT] 连接超时，将在后台重试...")
            return False
        except Exception as e:
            print(f"[MQTT] 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False
        print("[MQTT] 已断开连接")

    @property
    def is_connected(self):
        return self._connected

    # ------------------------------------------------------------------
    # 指令发送
    # ------------------------------------------------------------------

    def send_command(self, action: str, **kwargs):
        """
        发送控制指令

        Args:
            action: 动作名称（如 'sit'）
            **kwargs: 附加参数（速度、位移等）
        """
        payload = {
            "action": action,
            "timestamp": time.monotonic() * 1000.0,  # monotonic 毫秒
            **kwargs,
        }
        msg = json.dumps(payload, ensure_ascii=False)

        result = self._client.publish(
            config.MQTT_TOPIC,
            msg,
            qos=config.MQTT_QOS,
        )

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"  [MQTT] 已发送: {msg}")
        else:
            print(f"  [MQTT] 发送失败 (rc={result.rc})")

    def send_sit(self, velocity: float = 0.0, displacement: float = 0.0):
        """便捷方法：发送坐下指令"""
        self.send_command(
            action="sit",
            velocity=round(velocity, 4),
            displacement=round(displacement, 4),
        )

    def send_stand(self, velocity: float = 0.0, displacement: float = 0.0):
        """便捷方法：发送站立指令"""
        self.send_command(
            action="stand",
            velocity=round(velocity, 4),
            displacement=round(displacement, 4),
        )

    def send_rotate(self, angle: float = 0.0):
        """便捷方法：发送旋转指令"""
        self.send_command(
            action="rotate",
            angle=round(angle, 1),
        )

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
        else:
            print(f"[MQTT] 连接失败，原因码: {reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected = False
        if reason_code != 0:
            print(f"[MQTT] 意外断开，原因码: {reason_code}")
