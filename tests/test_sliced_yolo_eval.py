import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.common.tiling import generate_slices
from scripts.eval.sliced_yolo_eval import class_aware_nms, match_predictions


class SlicedYoloEvalTest(unittest.TestCase):
    def test_generate_slices_covers_image_edges(self):
        slices = generate_slices(width=1920, height=1080, slice_size=896, overlap=0.25)

        self.assertEqual(slices[0], (0, 0, 896, 896))
        self.assertTrue(any(x2 == 1920 for _, _, x2, _ in slices))
        self.assertTrue(any(y2 == 1080 for _, _, _, y2 in slices))

    def test_class_aware_nms_keeps_overlapping_different_classes(self):
        preds = np.array(
            [
                [0, 0, 100, 100, 0.9, 0],
                [5, 5, 105, 105, 0.8, 0],
                [5, 5, 105, 105, 0.7, 1],
            ],
            dtype=np.float32,
        )

        kept = class_aware_nms(preds, iou_thresh=0.5)

        self.assertEqual(len(kept), 2)
        self.assertEqual(set(kept[:, 5].astype(int)), {0, 1})

    def test_match_predictions_matches_once_per_iou_threshold(self):
        preds = np.array(
            [
                [0, 0, 100, 100, 0.9, 0],
                [0, 0, 100, 100, 0.8, 0],
            ],
            dtype=np.float32,
        )
        targets = np.array([[0, 0, 100, 100, 0]], dtype=np.float32)
        iouv = np.array([0.5, 0.75], dtype=np.float32)

        tp = match_predictions(preds, targets, iouv)

        self.assertEqual(tp.shape, (2, 2))
        self.assertEqual(tp[0].tolist(), [True, True])
        self.assertEqual(tp[1].tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
