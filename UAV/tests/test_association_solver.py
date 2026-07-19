import unittest

import numpy as np

from src.solver.sca_fp import SCAFPConfig, SCAFPOptimizer


class AssociationSolverTest(unittest.TestCase):
    def _solver(self, *, num_uavs=2, num_users=1, load_cap=1):
        return SCAFPOptimizer(
            SCAFPConfig(),
            M=num_uavs,
            K=num_users,
            T=1,
            p_max=1.0,
            noise_power=1.0,
            load_cap=load_cap,
        )

    def test_inactive_zero_power_does_not_lock_previous_association(self):
        solver = self._solver()
        gains = np.array([[1.0], [10.0]], dtype=np.float64)

        association = solver._optimize_association(
            channel_gains=gains,
            user_weights=np.ones(1),
        )

        np.testing.assert_array_equal(
            association,
            np.array([[0.0], [1.0]], dtype=np.float32),
        )

    def test_capacity_constrained_assignment_serves_every_user_once(self):
        solver = self._solver(num_uavs=2, num_users=3, load_cap=2)
        gains = np.array(
            [
                [10.0, 9.0, 8.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )

        association = solver._optimize_association(
            channel_gains=gains,
            user_weights=np.ones(3),
        )

        np.testing.assert_array_equal(association.sum(axis=0), np.ones(3))
        self.assertTrue(np.all(association.sum(axis=1) <= 2))
        self.assertEqual(float(association[0].sum()), 2.0)

    def test_infeasible_total_capacity_is_rejected(self):
        solver = self._solver(num_uavs=2, num_users=5, load_cap=2)

        with self.assertRaisesRegex(ValueError, "capacity is infeasible"):
            solver._optimize_association(
                channel_gains=np.ones((2, 5)),
                user_weights=np.ones(5),
            )


if __name__ == "__main__":
    unittest.main()
