import logging
import os
import threading
import time

import cv2

log = logging.getLogger(__name__)

# Force TCP transport — UDP drops cause frequent frame corruption on RTSP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class FrameGrabber:
    """Background RTSP reader.

    OpenCV's RTSP backend buffers decoded frames without a reliable way
    to cap the buffer (CAP_PROP_BUFFERSIZE is ignored by most FFmpeg
    builds). If the main loop only reads every N seconds, it ends up
    consuming frames that are many seconds old.

    This class runs a dedicated thread that reads frames at the stream's
    native rate, always keeping just the most recent one in memory.
    grab() returns whatever's freshest.
    """

    # Reject frames older than this; forces reconnect if stream stalls.
    _STALE_SECONDS = 5.0

    def __init__(self, url: str):
        self._url = url
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: tuple[float, object] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, name="rtsp-reader", daemon=True,
        )
        self._thread.start()

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

    def _read_loop(self) -> None:
        while self._running:
            try:
                if self._cap is None:
                    self._open()
                assert self._cap is not None
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    log.warning("read failed, reconnecting")
                    self._close()
                    time.sleep(1)
                    continue
                with self._lock:
                    self._latest = (time.monotonic(), frame)
            except Exception as e:
                log.warning("reader error: %s", e)
                self._close()
                time.sleep(2)

    def grab(self):
        """Return the most recent frame. Blocks briefly if none yet."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                snapshot = self._latest
            if snapshot is not None:
                ts, frame = snapshot
                age = time.monotonic() - ts
                if age < self._STALE_SECONDS:
                    return frame
            time.sleep(0.1)
        raise RuntimeError("no fresh frame available within 10s")

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close()
