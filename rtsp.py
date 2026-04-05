import logging
import os
import time

import cv2

log = logging.getLogger(__name__)

# Force TCP transport — UDP drops cause frequent frame corruption on RTSP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class FrameGrabber:
    """Persistent RTSP capture with automatic reconnect.

    We always want the newest frame, not the one at the head of the
    decoder buffer, so each grab() flushes pending frames before
    retrieving.
    """

    def __init__(self, url: str):
        self._url = url
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> None:
        log.info("opening RTSP stream")
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("failed to open RTSP stream")
        self._cap = cap

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def grab(self):
        for attempt in range(3):
            if self._cap is None:
                try:
                    self._open()
                except Exception as e:
                    log.warning("open failed: %s", e)
                    time.sleep(2 ** attempt)
                    continue

            assert self._cap is not None
            for _ in range(4):
                self._cap.grab()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                log.warning("read failed, reconnecting")
                self._close()
                time.sleep(1)
                continue
            return frame

        raise RuntimeError("failed to grab frame after retries")

    def close(self) -> None:
        self._close()
