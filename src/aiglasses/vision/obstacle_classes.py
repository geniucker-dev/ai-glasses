from __future__ import annotations

YOLOE_OBSTACLE_CLASS_NAMES = [
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "animal",
    "scooter",
    "stroller",
    "dog",
    "pole",
    "post",
    "column",
    "pillar",
    "stanchion",
    "bollard",
    "utility pole",
    "telegraph pole",
    "light pole",
    "street pole",
    "signpost",
    "support post",
    "vertical post",
    "bench",
    "chair",
    "potted plant",
    "hydrant",
    "cone",
    "stone",
    "box",
]

YOLOE_OBSTACLE_CLASS_ID_NAMES = dict(enumerate(YOLOE_OBSTACLE_CLASS_NAMES))
OBSTACLE_LABELS = set(YOLOE_OBSTACLE_CLASS_NAMES)
