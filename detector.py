import logging

from ultralytics import YOLO

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO


class PersonDetector:
    """Stock YOLO person detector.

    The generic COCO 'person' class is not ideal for a swaddled baby
    viewed top-down — if accuracy is lacking, replace this module with
    a small binary classifier fine-tuned on frames from FRAME_LOG_DIR.
    The rest of the pipeline only depends on the detect() signature.
    """

    def __init__(self, model_path: str, confidence: float):
        log.info("loading model %s", model_path)
        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self, frame) -> tuple[bool, float]:
        """Return (person_present, best_confidence)."""
        results = self._model.predict(
            frame,
            conf=self._confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        best = 0.0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best:
                    best = c
        return best > 0.0, best
