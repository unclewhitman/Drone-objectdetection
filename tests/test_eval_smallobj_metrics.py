import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval.eval_smallobj_metrics import filter_small_gt


def test_filter_small_gt():
    gts = [(0, 0.5, 0.5, 0.02, 0.02), (1, 0.5, 0.5, 0.5, 0.5)]
    small = filter_small_gt(gts, area_ratio_thresh=0.01)
    assert len(small) == 1
