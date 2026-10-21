"""Numerical engineering qualification; these tests are not efficacy evidence."""

import unittest

import torch
from models import MODELS, V2_HIDDEN, make_model, matched_hidden_sizes, parameter_count


class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(131)
        self.history = torch.randn(3, 8, 5, 4) * 0.1

    def test_candidate_shape_finite_and_backward(self):
        model = make_model("radial_mixture", 64, 8)
        prediction, auxiliary = model(self.history, 6)
        self.assertEqual(tuple(prediction.shape), (3, 6, 5, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        loss = prediction.square().mean() + auxiliary
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_candidate_is_object_permutation_equivariant(self):
        model = make_model("radial_mixture", 64, 8).eval()
        permutation = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            original, _ = model(self.history, 4)
            permuted, _ = model(self.history[:, :, permutation], 4)
        self.assertTrue(
            torch.allclose(permuted, original[:, :, permutation], atol=2e-6, rtol=2e-5)
        )

    def test_candidate_is_axis_permutation_equivariant(self):
        model = make_model("radial_mixture", 64, 8).eval()
        swapped_history = self.history[..., [1, 0, 3, 2]]
        with torch.no_grad():
            original, _ = model(self.history, 4)
            swapped, _ = model(swapped_history, 4)
        self.assertTrue(
            torch.allclose(
                swapped,
                original[..., [1, 0, 3, 2]],
                atol=2e-6,
                rtol=2e-5,
            )
        )

    def test_zero_initialized_candidate_is_constant_velocity(self):
        model = make_model("radial_mixture", 64, 8).eval()
        with torch.no_grad():
            prediction, _ = model(self.history, 3)
        current = self.history[:, -1]
        expected = []
        for step in range(1, 4):
            state = current.clone()
            state[..., :2] = current[..., :2] + 0.05 * step * current[..., 2:]
            expected.append(state)
        self.assertTrue(
            torch.allclose(prediction, torch.stack(expected, dim=1), atol=1e-7)
        )

    def test_pair_forces_conserve_total_momentum(self):
        model = make_model("radial_mixture", 64, 8).eval()
        with torch.no_grad():
            torch.nn.init.normal_(model.force_expert_a[-1].weight, std=0.2)
            torch.nn.init.normal_(model.force_expert_b[-1].weight, std=0.2)
            model(self.history, 3)
        diagnostics = model.diagnostics()
        self.assertLess(diagnostics["pair_momentum_residual_max"], 2e-5)

    def test_learned_and_fixed_gate_contract(self):
        learned = make_model("radial_mixture", 64, 8).eval()
        fixed = make_model("radial_fixed", 64, 8).eval()
        with torch.no_grad():
            learned(self.history, 3)
            fixed(self.history, 3)
        learned_stats = learned.diagnostics()
        fixed_stats = fixed.diagnostics()
        self.assertGreaterEqual(learned_stats["gate_min"], 0.05)
        self.assertLessEqual(learned_stats["gate_max"], 0.95)
        self.assertEqual(fixed_stats["gate_mean"], 0.5)
        self.assertEqual(parameter_count(learned), parameter_count(fixed))

    def test_all_models_share_contract(self):
        widths = matched_hidden_sizes(8, 64)
        for name in MODELS:
            with self.subTest(name=name):
                model = make_model(name, widths[name], 8).eval()
                with torch.no_grad():
                    output, auxiliary = model(self.history[:1], 2)
                self.assertEqual(tuple(output.shape), (1, 2, 5, 4))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(auxiliary))

    def test_width_matching_is_optimal_on_search_grid(self):
        widths = matched_hidden_sizes(8, 64)
        target = parameter_count(make_model("radial_mixture", 64, 8))
        for name in MODELS[3:]:
            chosen = abs(parameter_count(make_model(name, widths[name], 8)) - target)
            alternatives = [
                abs(parameter_count(make_model(name, width, 8)) - target)
                for width in range(32, 133, 4)
            ]
            self.assertEqual(chosen, min(alternatives))

    def test_predecessor_comparator_width_is_frozen(self):
        widths = matched_hidden_sizes(8, 64)
        self.assertEqual(widths["adaptive_context_v2"], V2_HIDDEN)
        with self.assertRaises(ValueError):
            make_model("adaptive_context_v2", V2_HIDDEN - 4, 8)

    def test_invalid_inputs_fail_closed(self):
        model = make_model("radial_mixture", 64, 8)
        with self.assertRaises(ValueError):
            model(self.history[:, :-1], 2)
        with self.assertRaises(ValueError):
            model(self.history, 0)
        broken = self.history.clone()
        broken[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            model(broken, 2)


if __name__ == "__main__":
    unittest.main()
