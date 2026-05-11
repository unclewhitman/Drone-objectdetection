import math
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.common.sobj_utils import xywh_to_xyxy, bbox_area, iou_xyxy


def test_xywh_to_xyxy():
    x, y, w, h = 0.5, 0.5, 0.2, 0.4
    x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
    assert x1 == 0.4 and y1 == 0.3 and x2 == 0.6 and y2 == 0.7


def test_bbox_area():
    assert math.isclose(bbox_area((0.1, 0.2, 0.3, 0.6)), 0.08)


def test_iou_xyxy():
    a = (0.0, 0.0, 0.5, 0.5)
    b = (0.25, 0.25, 0.75, 0.75)
    iou = iou_xyxy(a, b)
    assert round(iou, 3) == 0.143
