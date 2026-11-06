"""Uses the existing real checkpoint, with synthetic joint feedback only."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / "dexscrew/xhand-deploy"
sys.path.insert(0, str(ROOT))
from nut_policy import NutPolicy, XHAND_TO_POLICY


class NutPolicyTests(unittest.TestCase):
    def test_equivalence_to_timed_baseline(self):
        torch.set_num_threads(1)
        previous_dir = Path.cwd()
        try:
            os.chdir(ROOT)
            # Original module loads its model but does not enter main/open any device.
            import xhand_deploy as baseline
            policy = NutPolicy()
            previous = np.array([-.17, 1.1, .4, 1.1, .4, 0, 0, 0, 0, 1.3, .6, .45])
            obs = torch.zeros(1, 96)
            for _ in range(40):
                measured = previous[baseline.POLICY_TO_XHAND_IDX]
                with torch.no_grad():
                    x = baseline.F.pad(torch.clamp(obs, -5, 5), (0, 96-obs.shape[1]))
                    x = torch.clamp((x-baseline.obs_mean)/torch.sqrt(baseline.obs_var+1e-5), -5, 5)
                    h = torch.clamp((baseline.proprio_hist_buffer-baseline.sa_mean)/torch.sqrt(baseline.sa_var+1e-5), -5, 5)
                    mu, _, _ = baseline.jit_model(dict(obs=x, proprio_hist=h, point_cloud_info=torch.zeros(1,100,3)))
                    action = torch.clamp(mu, -1, 1).numpy().flatten()
                    action[5:9] = 0
                    previous = np.clip(previous + action*baseline.ACTION_SCALE,
                                       baseline.xhand_dof_lower_limits, baseline.xhand_dof_upper_limits)
                    obs = baseline.proprio_hist_buffer[:, -3:, :].reshape(1,-1).clone()
                    baseline.update_proprio_hist(measured[XHAND_TO_POLICY], previous)
                actual = policy.step(measured)
                np.testing.assert_allclose(actual, previous[baseline.POLICY_TO_XHAND_IDX], atol=1e-7, rtol=0)
                torch.testing.assert_close(policy.history, baseline.proprio_hist_buffer, atol=1e-7, rtol=0)
                torch.testing.assert_close(policy.obs, obs, atol=1e-7, rtol=0)
            with self.assertRaises(ValueError):
                policy.step(np.full(12, np.nan))
            policy.reset()
            np.testing.assert_allclose(policy.initial_hardware, policy.previous_target[baseline.POLICY_TO_XHAND_IDX])
        finally:
            os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
