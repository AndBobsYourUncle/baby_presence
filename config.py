import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str | None
    mqtt_pass: str | None
    mqtt_client_id: str
    base_topic: str
    discovery_prefix: str
    device_id: str
    device_name: str
    model_path: str
    confidence: float
    sample_interval: float
    debounce_seconds: float
    frame_log_dir: str | None
    frame_log_interval: float


def _req(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} environment variable is required")
    return v


def load() -> Config:
    return Config(
        rtsp_url=_req("RTSP_URL"),
        mqtt_host=_req("MQTT_HOST"),
        mqtt_port=int(os.environ.get("MQTT_PORT", "1883")),
        mqtt_user=os.environ.get("MQTT_USER") or None,
        mqtt_pass=os.environ.get("MQTT_PASS") or None,
        mqtt_client_id=os.environ.get("MQTT_CLIENT_ID", "baby_presence"),
        base_topic=os.environ.get("BASE_TOPIC", "babypresence"),
        discovery_prefix=os.environ.get("HA_DISCOVERY_PREFIX", "homeassistant"),
        device_id=os.environ.get("DEVICE_ID", "baby_presence_crib"),
        device_name=os.environ.get("DEVICE_NAME", "Baby Crib"),
        model_path=os.environ.get("MODEL_PATH", "yolov8n.pt"),
        confidence=float(os.environ.get("CONFIDENCE", "0.25")),
        sample_interval=float(os.environ.get("SAMPLE_INTERVAL", "2.0")),
        debounce_seconds=float(os.environ.get("DEBOUNCE_SECONDS", "15.0")),
        frame_log_dir=os.environ.get("FRAME_LOG_DIR") or None,
        frame_log_interval=float(os.environ.get("FRAME_LOG_INTERVAL", "60.0")),
    )
