"""Adversarial tests for adequacy, statistics, split separation and frozen intake."""

import unittest

from control import ROOT, adequate, holm, read, sign_permutation, summarize


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.p = read(ROOT / "protocol.json")
        self.curve = [
            {"step": s, "horizon": 64, "train_rmse": 0.1, "validation_rmse": 0.12}
            for s in range(3500, 6500, 500)
        ]

    def test_poor_plateau_cannot_pass(self):
        self.assertFalse(adequate(self.curve, 1, 1, self.p)["pass"])

    def test_fitting_and_stability_required(self):
        self.assertTrue(adequate(self.curve, 0.1, 1, self.p)["pass"])
        self.curve[-1]["validation_rmse"] = 0.10
        self.assertFalse(adequate(self.curve, 0.1, 1, self.p)["pass"])

    def test_training_still_improving_fails(self):
        self.curve[-1]["train_rmse"] = 0.08
        self.assertFalse(adequate(self.curve, 0.08, 1, self.p)["pass"])

    def test_short_curriculum_cannot_pass(self):
        for row in self.curve:
            row["horizon"] = 8
        self.assertFalse(adequate(self.curve, 0.1, 1, self.p)["pass"])

    def test_minimum_steps(self):
        self.assertFalse(adequate(self.curve[:-1], 0.1, 1, self.p)["pass"])

    def test_nonfinite_fails(self):
        self.curve[-1]["train_rmse"] = float("nan")
        self.assertFalse(adequate(self.curve, 0.1, 1, self.p)["pass"])

    def test_exact_six_unit_probability(self):
        self.assertEqual(sign_permutation([1] * 6), 1 / 64)
        self.assertEqual(sign_permutation([0] * 6), 1)
        self.assertEqual(sign_permutation([-1] * 6), 1)

    def test_holm(self):
        self.assertEqual(holm([1 / 64, 1 / 64]), [1 / 32, 1 / 32])
        self.assertEqual(holm([0.04, 0.02]), [0.04, 0.04])

    def test_split_seeds_are_disjoint_from_old_study(self):
        seeds = []
        for b in (
            self.p["data_blocks"]
            + self.p["calibration_blocks"]
            + [3100001, 3400001, 3700001, 4000001]
        ):
            for domain_offset in (0, 50):
                seeds.extend([b * 100 + domain_offset + 1, b * 100 + domain_offset + 2])
                seeds.extend(
                    b * 100 + domain_offset + 3 + regime for regime in (0, 10, 20, 30)
                )
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertLess(max(seeds), 2**32)

    def rows(self):
        rows = []
        for d in self.p["domains"]:
            for b in self.p["data_blocks"]:
                for s in self.p["model_seeds"]:
                    for m in self.p["models"]:
                        rows.append(
                            {
                                "domain": d,
                                "block": b,
                                "seed": s,
                                "model": m,
                                "adequacy": {"pass": True},
                                "metrics": {
                                    r: {
                                        "position_rmse": 0.8
                                        if m == "radial_mixture"
                                        else 1.0
                                    }
                                    for r in self.p["regimes"]
                                },
                                "intervention": {
                                    r: {"position_rmse": 1.0} for r in self.p["regimes"]
                                },
                            }
                        )
        return rows

    def test_complete_positive_fixture(self):
        result = summarize(self.rows(), self.p)
        self.assertEqual(result["decision"], "ADAPTIVE_BENEFIT_SUPPORTED_IN_THIS_STUDY")
        self.assertTrue(result["scene_dependent_mechanism_supported"])

    def test_missing_and_duplicate_cells(self):
        rows = self.rows()
        self.assertEqual(summarize(rows[:-1], self.p)["decision"], "INCOMPLETE")
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], self.p)

    def test_weak_fixed_gain_not_rescued_by_baseline_win(self):
        rows = self.rows()
        for row in rows:
            if row["model"] == "radial_fixed":
                for regime in self.p["regimes"]:
                    row["metrics"][regime]["position_rmse"] = 0.805
        self.assertEqual(
            summarize(rows, self.p)["decision"], "ADAPTIVE_BENEFIT_NOT_ESTABLISHED"
        )

    def test_adaptation_and_architecture_are_separate(self):
        rows = self.rows()
        for row in rows:
            for regime in self.p["regimes"]:
                row["intervention"][regime]["position_rmse"] = 0.8
        result = summarize(rows, self.p)
        self.assertTrue(result["decision"].startswith("ADAPTIVE_BENEFIT_SUPPORTED"))
        self.assertFalse(result["scene_dependent_mechanism_supported"])

    def test_undertrained_test_cannot_be_reported(self):
        rows = self.rows()
        rows[0]["adequacy"]["pass"] = False
        with self.assertRaises(ValueError):
            summarize(rows, self.p)


if __name__ == "__main__":
    unittest.main()
