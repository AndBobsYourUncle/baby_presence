import logging
import os
import signal
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

from config import load  # noqa: E402 — must run after load_dotenv
from detector import PersonDetector
from mqtt_client import MQTTPublisher
from rtsp import FrameGrabber

log = logging.getLogger("baby_presence")


class Debouncer:
    """Reports a state change only after it has held for `hold_seconds`.

    Prevents flicker from a parent reaching in, a brief mis-detection,
    or a single frame of confusion.
    """

    def __init__(self, hold_seconds: float, initial: bool = False):
        self._hold = hold_seconds
        self._published = initial
        self._candidate = initial
        self._candidate_since = time.monotonic()

    def update(self, observed: bool) -> bool | None:
        now = time.monotonic()
        if observed != self._candidate:
            self._candidate = observed
            self._candidate_since = now
            return None
        if (
            observed != self._published
            and (now - self._candidate_since) >= self._hold
        ):
            self._published = observed
            return observed
        return None


def _maybe_log_frame(
    frame,
    directory: Path,
    last_logged: float,
    interval: float,
    label: str,
) -> float:
    now = time.monotonic()
    if now - last_logged < interval:
        return last_logged
    # Best-effort: if the frame dir is on a flaky NAS mount, we don't
    # want a write failure to spam the main loop. Return `now` either
    # way so the next attempt waits a full interval.
    try:
        directory.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(directory / f"{ts}_{label}.jpg"), frame)
    except OSError as e:
        log.warning("frame log write failed: %s", e)
    return now


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load()

    grabber = FrameGrabber(cfg.rtsp_url)
    detector = PersonDetector(cfg.model_path, cfg.confidence)
    publisher = MQTTPublisher(
        host=cfg.mqtt_host,
        port=cfg.mqtt_port,
        user=cfg.mqtt_user,
        password=cfg.mqtt_pass,
        client_id=cfg.mqtt_client_id,
        base_topic=cfg.base_topic,
        discovery_prefix=cfg.discovery_prefix,
        device_id=cfg.device_id,
        device_name=cfg.device_name,
    )
    publisher.start()

    debouncer = Debouncer(cfg.debounce_seconds, initial=False)
    publisher.publish_state(False)

    frame_log_dir = Path(cfg.frame_log_dir) if cfg.frame_log_dir else None
    last_logged = 0.0

    stop = False

    def _handle_signal(signum, _frame):
        nonlocal stop
        log.info("received signal %s, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("baby_presence running")
    while not stop:
        loop_start = time.monotonic()
        try:
            frame = grabber.grab()
            present, conf = detector.detect(frame)
            log.debug("detection present=%s conf=%.2f", present, conf)

            changed = debouncer.update(present)
            if changed is not None:
                log.info("state -> %s", "ON" if changed else "OFF")
                publisher.publish_state(changed)

            if frame_log_dir is not None:
                label = f"{'occupied' if present else 'empty'}_c{conf:.2f}"
                last_logged = _maybe_log_frame(
                    frame,
                    frame_log_dir,
                    last_logged,
                    cfg.frame_log_interval,
                    label,
                )
        except Exception as e:
            log.exception("loop error: %s", e)
            time.sleep(2)

        elapsed = time.monotonic() - loop_start
        remaining = cfg.sample_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    grabber.close()
    publisher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
