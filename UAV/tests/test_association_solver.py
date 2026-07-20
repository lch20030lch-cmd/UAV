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
        with self.assertRaisesRegex(ValueError, "capacity is infeasible"):
            self._solver(num_uavs=2, num_users=5, load_cap=2)

    def test_warmstart_association_is_capacity_constrained(self):
        solver = self._solver(num_uavs=2, num_users=3, load_cap=2)
        env = {
            "q_current": np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]]),
            "user_positions": np.zeros((3, 2)),
            "target_positions": np.zeros((1, 2)),
            "channel_gains": np.ones((2, 3)),
            "user_weights": np.ones(3),
        }
        warm = {
            "delta_q": np.zeros((2, 3)),
            "delta_a": np.array([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]]),
            "delta_p": np.ones((2, 4)),
        }

        _, association, communication_power, _ = solver._warmstart_to_init(
            warm, solver._validate_environment(env)
        )

        np.testing.assert_array_equal(association.sum(axis=0), np.ones(3))
        self.assertTrue(np.all(association.sum(axis=1) <= 2))
        self.assertEqual(float(communication_power[association < 0.5].sum()), 0.0)

    def test_warmstart_a_and_p_reach_first_deployment_update(self):
        class CapturingSolver(SCAFPOptimizer):
            captured = None

            def _optimize_deployment_sca(self, Q, A, P_comm, P_sense, environment):
                self.captured = (A.copy(), P_comm.copy(), P_sense.copy())
                return Q.copy()

        solver = CapturingSolver(
            SCAFPConfig(max_outer_iters=1, max_inner_iters=1),
            M=2,
            K=2,
            T=1,
            p_max=1.0,
            noise_power=1.0,
            load_cap=1,
        )
        env = {
            "q_current": np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]]),
            "user_positions": np.array([[0.0, 0.0], [100.0, 0.0]]),
            "target_positions": np.array([[50.0, 0.0]]),
            "channel_gains": np.ones((2, 2)),
            "user_weights": np.ones(2),
        }
        warm = {
            "delta_q": np.zeros((2, 3)),
            "delta_a": np.eye(2),
            "delta_p": np.array([[0.6, 0.0, 0.4], [0.0, 0.7, 0.3]]),
        }

        solver.solve(env, warm_start=warm, seed=1)

        captured_a, captured_comm, captured_sense = solver.captured
        np.testing.assert_array_equal(captured_a, warm["delta_a"])
        np.testing.assert_allclose(captured_comm, warm["delta_p"][:, :2])
        np.testing.assert_allclose(captured_sense, warm["delta_p"][:, 2])

    def test_geometry_dependent_channel_changes_after_q_move(self):
        solver = self._solver(num_uavs=2, num_users=1, load_cap=1)
        env = solver._validate_environment({
            "q_current": np.array([[0.0, 0.0, 100.0], [500.0, 0.0, 100.0]]),
            "user_positions": np.array([[0.0, 0.0]]),
            "target_positions": np.array([[0.0, 0.0]]),
            "channel_gains": np.ones((2, 1)),
            "user_weights": np.ones(1),
        })
        moved = env["q_current"].copy()
        moved[0, 0] = 15.0

        original_gain = solver._communication_gains(env["q_current"], env)[0, 0]
        moved_gain = solver._communication_gains(moved, env)[0, 0]

        self.assertLess(moved_gain, original_gain)

    def test_deployment_projection_enforces_minimum_separation(self):
        solver = SCAFPOptimizer(
            SCAFPConfig(min_separation_m=10.0),
            M=2,
            K=1,
            T=1,
            load_cap=1,
            v_max=15.0,
        )
        current = np.array([[100.0, 100.0, 100.0], [105.0, 100.0, 100.0]])

        projected = solver._project_deployment_feasible(current.copy(), current)

        self.assertGreaterEqual(
            float(np.linalg.norm(projected[0] - projected[1])), 10.0 - 1e-5
        )
        self.assertTrue(
            np.all(np.linalg.norm(projected - current, axis=1) <= 15.0 + 1e-6)
        )


if __name__ == "__main__":
    unittest.main()
