import unittest
from types import SimpleNamespace

import numpy as np

from src.data.prompt_builder import (
    build_indexed_association_str,
    build_multimodal_prompt,
)


class IndexedAssociationPromptTest(unittest.TestCase):
    def _environment(self):
        return SimpleNamespace(
            q_current=np.array(
                [[0.0, 0.0, 100.0], [100.0, 100.0, 120.0]],
                dtype=np.float32,
            ),
            u_positions=np.array(
                [[10.0, 20.0], [80.0, 90.0]],
                dtype=np.float32,
            ),
            s_positions=np.array([[50.0, 50.0]], dtype=np.float32),
            association=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            user_weights=np.array([1.25, 0.75], dtype=np.float32),
            channel_gains_users=np.array(
                [[1.0, 4.0], [10.0, 2.0]],
                dtype=np.float32,
            ),
            comm_summary={
                "per_user_sinr_db": [12.5, 8.0],
                "per_uav_load": [1, 1],
                "rate_pressure": [0.5, 0.8],
            },
            sensing_summary={
                "per_target_sinr_db": [10.0],
                "localization_difficulty": [0.5],
                "uncovered_targets": 0,
                "best_uav_per_target": [0],
            },
        )

    def test_map_preserves_user_column_ids_and_channel_rank(self):
        text = build_indexed_association_str(self._environment())

        self.assertIn("columns follow user IDs u0..", text)
        self.assertIn("u0:xy=[10.0, 20.0]", text)
        self.assertIn("w=1.25", text)
        self.assertIn("best_sinr_db=12.5", text)
        self.assertIn("rank=m1>m0", text)
        self.assertIn("u1:xy=[80.0, 90.0]", text)
        self.assertIn("rank=m0>m1", text)

    def test_multimodal_prompt_contains_index_map_before_image_description(self):
        config = {
            "num_uavs": 2,
            "num_users": 2,
            "num_targets": 1,
            "area_size": [100.0, 100.0],
        }

        prompt = build_multimodal_prompt(self._environment(), config)

        self.assertLess(
            prompt.index("[Indexed Association Map]"),
            prompt.index("[Bird's-Eye-View Image]"),
        )

    def test_misaligned_channel_matrix_is_rejected(self):
        environment = self._environment()
        environment.channel_gains_users = np.ones((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "aligned with users"):
            build_indexed_association_str(environment)


if __name__ == "__main__":
    unittest.main()
