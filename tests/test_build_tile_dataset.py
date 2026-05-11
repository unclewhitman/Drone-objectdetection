import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data.build_tile_dataset import clip_labels_to_tile


class BuildTileDatasetTest(unittest.TestCase):
    def test_clip_labels_to_tile_rebases_coordinates(self):
        labels = [(3, 0.5, 0.5, 0.2, 0.2)]

        clipped = clip_labels_to_tile(
            labels,
            image_width=1000,
            image_height=1000,
            tile=(250, 250, 750, 750),
            min_visible=0.3,
        )

        self.assertEqual(len(clipped), 1)
        cls_id, x, y, w, h = clipped[0]
        self.assertEqual(cls_id, 3)
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.4)

    def test_clip_labels_to_tile_drops_mostly_invisible_box(self):
        labels = [(3, 0.1, 0.5, 0.2, 0.2)]

        clipped = clip_labels_to_tile(
            labels,
            image_width=1000,
            image_height=1000,
            tile=(150, 250, 650, 750),
            min_visible=0.3,
            center_required=False,
        )

        self.assertEqual(clipped, [])


if __name__ == "__main__":
    unittest.main()
