import json
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import mobilenet_v3_small

log = logging.getLogger(__name__)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _ClassifierDetector:
    """Fine-tuned binary occupancy classifier (MobileNetV3-small).

    Selected when a sidecar `.json` (written by scripts/train.py)
    exists next to the model file and declares a supported
    architecture.
    """

    def __init__(self, model_path: str, confidence: float, meta: dict):
        if meta["architecture"] != "mobilenet_v3_small":
            raise ValueError(
                f"unsupported architecture: {meta['architecture']}"
            )
        self._classes: list[str] = meta["classes"]
        self._occupied_idx = self._classes.index("occupied")
        self._threshold = confidence
        self._device = _pick_device()
        log.info("classifier device: %s", self._device)

        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(self._classes))
        state = torch.load(
            model_path, map_location=self._device, weights_only=True,
        )
        model.load_state_dict(state)
        model.to(self._device).eval()
        self._model = model

        self._transform = T.Compose([
            T.Resize((meta["input_size"], meta["input_size"])),
            T.ToTensor(),
            T.Normalize(mean=meta["normalize_mean"], std=meta["normalize_std"]),
        ])

    def detect(self, frame) -> tuple[bool, float]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)
            p = probs[0, self._occupied_idx].item()
        return p >= self._threshold, p


class _YoloDetector:
    """Stock YOLO person detector (COCO class 0).

    Used as a fallback when no classifier sidecar metadata is present.
    Requires ultralytics to be installed (not in requirements.txt by
    default — install with `pip install ultralytics` if you need it).
    """

    _PERSON_CLASS_ID = 0

    def __init__(self, model_path: str, confidence: float):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLO models. "
                "Install with: pip install ultralytics"
            ) from e
        log.info("loading YOLO model %s", model_path)
        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self, frame) -> tuple[bool, float]:
        results = self._model.predict(
            frame, conf=self._confidence,
            classes=[self._PERSON_CLASS_ID], verbose=False,
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


def PersonDetector(model_path: str, confidence: float):
    """Pick the detector implementation based on the model file.

    If a `.json` sidecar with an `architecture` field exists next to the
    model, use the classifier. Otherwise fall back to YOLO.
    """
    log.info("loading model %s", model_path)
    meta_path = Path(model_path).with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("architecture"):
            return _ClassifierDetector(model_path, confidence, meta)
    return _YoloDetector(model_path, confidence)
