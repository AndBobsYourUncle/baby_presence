import json
import logging

import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)


class MQTTPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
        client_id: str,
        base_topic: str,
        discovery_prefix: str,
        device_id: str,
        device_name: str,
    ):
        self._host = host
        self._port = port
        self._device_id = device_id
        self._device_name = device_name

        self._state_topic = f"{base_topic}/state"
        self._avail_topic = f"{base_topic}/availability"
        self._discovery_topic = (
            f"{discovery_prefix}/binary_sensor/{device_id}/config"
        )

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        if user:
            self._client.username_pw_set(user, password or "")
        # Last-will so HA shows the device as unavailable if we die.
        self._client.will_set(self._avail_topic, "offline", retain=True)
        self._client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            log.error("mqtt connect failed: %s", reason_code)
            return
        log.info("mqtt connected")
        self._publish_discovery()
        client.publish(self._avail_topic, "online", retain=True)

    def _publish_discovery(self) -> None:
        payload = {
            "name": "Baby Presence",
            "unique_id": self._device_id,
            "state_topic": self._state_topic,
            "availability_topic": self._avail_topic,
            "device_class": "occupancy",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device": {
                "identifiers": [self._device_id],
                "name": self._device_name,
                "model": "RTSP + YOLO",
                "manufacturer": "DIY",
            },
        }
        self._client.publish(
            self._discovery_topic,
            json.dumps(payload),
            retain=True,
        )

    def start(self) -> None:
        self._client.connect_async(self._host, self._port, keepalive=30)
        self._client.loop_start()

    def publish_state(self, present: bool) -> None:
        self._client.publish(
            self._state_topic,
            "ON" if present else "OFF",
            retain=True,
        )

    def stop(self) -> None:
        try:
            self._client.publish(self._avail_topic, "offline", retain=True)
        finally:
            self._client.loop_stop()
            self._client.disconnect()
