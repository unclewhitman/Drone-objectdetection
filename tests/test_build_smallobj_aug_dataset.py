import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data.build_smallobj_aug_dataset import select_small_objects


def test_select_small_objects():
    labels = [
        (0, 0.5, 0.5, 0.02, 0.02),
        (1, 0.5, 0.5, 0.5, 0.5),
    ]
    small = select_small_objects(labels, area_ratio_thresh=0.01)
    assert len(small) == 1
