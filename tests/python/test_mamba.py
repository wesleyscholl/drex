"""
tests.python.test_mamba — 100% coverage tests for drex.models.mamba.

Covers:
    _dt_init              — timescale bias initialisation
    MambaSSM              — construction, parameters, forward shapes, gradients,
                            D-skip, conv causality, sequential scan recurrence
    MambaLayer (use_l2)   — construction, gate parameter, forward, state update
    MambaLayer (no l2)    — forward, state passthrough
    DrexTransformer       — use_mamba=True integration, forward shapes,
                            trainable parameter count vs baseline
    train.py              — use_mamba config fields wired correctly
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from drex.models.mamba import MambaLayer, MambaSSM, _DT_MAX, _DT_MIN, _dt_init
from drex.models.memory import LayerState, MemoryState
from drex.models.transformer import DrexConfig, DrexTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zeros_state(batch: int, n_heads: int, d_k: int, device: torch.device) -> MemoryState:
    return MemoryState(
        M=torch.zeros(batch, n_heads, d_k, d_k, device=device),
        z=torch.zeros(batch, n_heads, d_k, device=device),
    )


def _small_ssm(**kwargs) -> MambaSSM:
    """MambaSSM with tiny dims for fast unit tests."""
    defaults = dict(d_model=16, d_state=4, d_conv=3, expand=2)
    defaults.update(kwargs)
    return MambaSSM(**defaults)


def _small_config(use_mamba: bool = True, use_l2: bool = True, **kwargs) -> DrexConfig:
    return DrexConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        vocab_size=64,
        max_seq_len=128,
        window_size=64,
        use_mamba=use_mamba,
        mamba_d_state=8,
        mamba_d_conv=3,
        mamba_expand=2,
        use_l2=use_l2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests for _dt_init
# ---------------------------------------------------------------------------


class TestDtInit(unittest.TestCase):
    def _make_linear(self, out_features: int = 64) -> nn.Linear:
        lin = nn.Linear(4, out_features, bias=True)
        nn.init.zeros_(lin.bias)
        return lin

    def test_softplus_of_bias_in_range(self) -> None:
        lin = self._make_linear(128)
        _dt_init(lin, dt_min=_DT_MIN, dt_max=_DT_MAX)
        sp = torch.nn.functional.softplus(lin.bias.detach())
        self.assertTrue((sp >= _DT_MIN * 0.99).all(), "Some Δ below dt_min")
        self.assertTrue((sp <= _DT_MAX * 1.01).all(), "Some Δ above dt_max")

    def test_no_reinit_attribute_set(self) -> None:
        lin = self._make_linear(32)
        _dt_init(lin)
        self.assertTrue(hasattr(lin.bias, "_no_reinit"))
        self.assertTrue(lin.bias._no_reinit)

    def test_custom_dt_range(self) -> None:
        lin = self._make_linear(64)
        _dt_init(lin, dt_min=0.01, dt_max=0.5)
        sp = torch.nn.functional.softplus(lin.bias.detach())
        self.assertTrue((sp >= 0.01 * 0.99).all())
        self.assertTrue((sp <= 0.5 * 1.01).all())

    def test_bias_values_differ(self) -> None:
        """Bias values should be drawn from a distribution, not all identical."""
        torch.manual_seed(0)
        lin = self._make_linear(64)
        _dt_init(lin)
        self.assertGreater(lin.bias.detach().std().item(), 0.0)


# ---------------------------------------------------------------------------
# Tests for MambaSSM — construction
# ---------------------------------------------------------------------------


class TestMambaSSMConstruction(unittest.TestCase):
    def test_default_construction(self) -> None:
        m = MambaSSM(d_model=64)
        self.assertEqual(m.d_inner, 128)   # expand=2
        self.assertEqual(m.d_state, 16)
        self.assertEqual(m.d_conv, 4)
        self.assertEqual(m.dt_rank, 4)     # max(1, 64 // 16)

    def test_small_d_model_dt_rank_minimum(self) -> None:
        m = MambaSSM(d_model=8, d_state=4)
        self.assertEqual(m.dt_rank, 1)     # max(1, 8 // 16) = 1

    def test_custom_dims(self) -> None:
        m = MambaSSM(d_model=32, d_state=8, d_conv=3, expand=4)
        self.assertEqual(m.d_inner, 128)
        self.assertEqual(m.d_state, 8)
        self.assertEqual(m.d_conv, 3)

    def test_in_proj_shape(self) -> None:
        m = _small_ssm()
        # d_model=16, d_inner=32 → in_proj: 16 → 2*d_inner=64
        self.assertEqual(m.in_proj.weight.shape, (64, 16))

    def test_conv1d_is_depthwise(self) -> None:
        m = _small_ssm()
        # groups == d_inner for depthwise
        self.assertEqual(m.conv1d.groups, m.d_inner)
        self.assertEqual(m.conv1d.in_channels, m.d_inner)
        self.assertEqual(m.conv1d.out_channels, m.d_inner)

    def test_x_proj_shape(self) -> None:
        m = _small_ssm()
        # d_inner → dt_rank + 2*d_state
        expected_out = m.dt_rank + 2 * m.d_state
        self.assertEqual(m.x_proj.weight.shape, (expected_out, m.d_inner))

    def test_dt_proj_shape(self) -> None:
        m = _small_ssm()
        self.assertEqual(m.dt_proj.weight.shape, (m.d_inner, m.dt_rank))
        self.assertEqual(m.dt_proj.bias.shape, (m.d_inner,))

    def test_log_A_shape(self) -> None:
        m = _small_ssm()
        self.assertEqual(m.log_A.shape, (m.d_inner, m.d_state))

    def test_log_A_positive_init(self) -> None:
        """log_A = log(i+1) so all values must be >= 0."""
        m = _small_ssm()
        self.assertTrue((m.log_A.detach() >= 0).all())

    def test_D_shape_and_ones_init(self) -> None:
        m = _small_ssm()
        self.assertEqual(m.D.shape, (m.d_inner,))
        self.assertTrue((m.D.detach() == 1.0).all())

    def test_out_proj_shape(self) -> None:
        m = _small_ssm()
        self.assertEqual(m.out_proj.weight.shape, (m.d_model, m.d_inner))

    def test_dt_bias_in_timescale_range(self) -> None:
        m = _small_ssm()
        sp = torch.nn.functional.softplus(m.dt_proj.bias.detach())
        self.assertTrue((sp >= _DT_MIN * 0.99).all())
        self.assertTrue((sp <= _DT_MAX * 1.01).all())

    def test_all_params_have_correct_device(self) -> None:
        m = _small_ssm()
        for name, p in m.named_parameters():
            self.assertEqual(p.device.type, "cpu", f"{name} on wrong device")


# ---------------------------------------------------------------------------
# Tests for MambaSSM — forward
# ---------------------------------------------------------------------------


class TestMambaSSMForward(unittest.TestCase):
    def setUp(self) -> None:
        self.m = _small_ssm()  # d_model=16, d_inner=32, d_state=4, d_conv=3
        self.B, self.S, self.D = 2, 8, 16

    def _x(self) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.randn(self.B, self.S, self.D)

    def test_output_shape(self) -> None:
        y = self.m(self._x())
        self.assertEqual(y.shape, (self.B, self.S, self.D))

    def test_output_is_finite(self) -> None:
        y = self.m(self._x())
        self.assertTrue(y.isfinite().all())

    def test_gradient_flows(self) -> None:
        x = self._x().requires_grad_(True)
        y = self.m(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.isfinite().all())

    def test_all_params_receive_gradient(self) -> None:
        x = self._x().requires_grad_(True)
        y = self.m(x)
        y.sum().backward()
        for name, p in self.m.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no gradient")

    def test_deterministic_given_seed(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(self.B, self.S, self.D)
        y1 = self.m(x)
        y2 = self.m(x)
        self.assertTrue(torch.allclose(y1, y2))

    def test_batch_independence(self) -> None:
        """Different batch items must not affect each other."""
        x = self._x()
        y_full = self.m(x)
        # Forward on item 0 alone
        y0 = self.m(x[:1])
        self.assertTrue(torch.allclose(y_full[:1], y0, atol=1e-5))

    def test_sequence_length_one(self) -> None:
        x = torch.randn(1, 1, self.D)
        y = self.m(x)
        self.assertEqual(y.shape, (1, 1, self.D))

    def test_longer_sequence(self) -> None:
        x = torch.randn(1, 64, self.D)
        y = self.m(x)
        self.assertEqual(y.shape, (1, 64, self.D))

    def test_training_vs_eval_consistent(self) -> None:
        """MambaSSM has no dropout, so train and eval should give same output."""
        x = self._x()
        self.m.eval()
        with torch.no_grad():
            y_eval = self.m(x)
        self.m.train()
        with torch.no_grad():
            y_train = self.m(x)
        self.assertTrue(torch.allclose(y_eval, y_train))

    def test_d_skip_contributes_to_output(self) -> None:
        """Setting D=0 should change the output vs D=1 (default)."""
        x = self._x()
        y_default = self.m(x).detach()
        # Zero out D skip
        with torch.no_grad():
            self.m.D.fill_(0.0)
        y_no_skip = self.m(x).detach()
        self.assertFalse(torch.allclose(y_default, y_no_skip))
        # Restore
        with torch.no_grad():
            self.m.D.fill_(1.0)

    def test_conv_causality_no_future_leakage(self) -> None:
        """
        Changing a future token should not affect the output at earlier positions.
        This validates the causal padding + right-trim in the conv branch.
        """
        torch.manual_seed(2)
        x1 = torch.randn(1, self.S, self.D)
        x2 = x1.clone()
        x2[:, -1] += 10.0   # modify only the last token

        with torch.no_grad():
            y1 = self.m(x1)
            y2 = self.m(x2)

        # First S-1 positions must be unaffected by the last token change
        self.assertTrue(torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5))


# ---------------------------------------------------------------------------
# Tests for MambaSSM — SSM scan recurrence
# ---------------------------------------------------------------------------


class TestMambaSSMScan(unittest.TestCase):
    def test_pure_zero_input_gives_zero_output_when_D_zeroed(self) -> None:
        """
        When x=0, x_branch=0 after silu, so ssm state stays 0 and y=0.
        D=0 removes the skip, z=silu(0)=0 kills the gate: output must be zero.
        """
        m = _small_ssm(d_model=16, d_state=4)
        with torch.no_grad():
            m.D.fill_(0.0)
            # in_proj: all zeros → x_branch=silu(0)=0, z_branch=0
            m.in_proj.weight.fill_(0.0)
        x = torch.zeros(1, 8, 16)
        y = m(x)
        self.assertTrue(torch.allclose(y, torch.zeros_like(y), atol=1e-6))

    def test_output_changes_with_input(self) -> None:
        m = _small_ssm()
        torch.manual_seed(3)
        x1 = torch.randn(1, 8, 16)
        x2 = torch.randn(1, 8, 16)
        y1, y2 = m(x1), m(x2)
        self.assertFalse(torch.allclose(y1, y2))

    def test_log_A_positive_enforces_stable_A_neg(self) -> None:
        """A_neg = -exp(log_A) must be negative for all elements."""
        m = _small_ssm()
        A_neg = -torch.exp(m.log_A.detach())
        self.assertTrue((A_neg < 0).all())

    def test_state_initialized_to_zero_each_forward(self) -> None:
        """Two consecutive calls with same input must give same output (no hidden state persistence)."""
        m = _small_ssm()
        torch.manual_seed(5)
        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            y1 = m(x)
            y2 = m(x)
        self.assertTrue(torch.allclose(y1, y2))


# ---------------------------------------------------------------------------
# Tests for MambaLayer — with L2 InfiniAttention
# ---------------------------------------------------------------------------


class TestMambaLayerWithL2(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 32
        self.n_heads = 4
        self.d_k = self.d_model // self.n_heads
        self.layer = MambaLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_state=8,
            d_conv=3,
            expand=2,
            use_l2=True,
        )
        self.B, self.S = 2, 12

    def _state(self) -> MemoryState:
        return _zeros_state(self.B, self.n_heads, self.d_k, torch.device("cpu"))

    def _x(self) -> torch.Tensor:
        torch.manual_seed(10)
        return torch.randn(self.B, self.S, self.d_model)

    def test_has_ssm_and_l2_and_gate(self) -> None:
        self.assertIsInstance(self.layer.ssm, MambaSSM)
        self.assertTrue(hasattr(self.layer, "l2"))
        self.assertTrue(hasattr(self.layer, "gate"))
        self.assertEqual(self.layer.gate.shape, (1,))

    def test_gate_init_zero(self) -> None:
        self.assertEqual(self.layer.gate.item(), 0.0)

    def test_output_shape(self) -> None:
        out, _ = self.layer(self._x(), self._state())
        self.assertEqual(out.shape, (self.B, self.S, self.d_model))

    def test_state_is_updated(self) -> None:
        state0 = self._state()
        x = self._x()
        _, state1 = self.layer(x, state0)
        # M and z should have been written to (non-zero after a non-zero input)
        self.assertFalse(torch.allclose(state1.M, state0.M))

    def test_output_is_finite(self) -> None:
        out, _ = self.layer(self._x(), self._state())
        self.assertTrue(out.isfinite().all())

    def test_gradient_flows(self) -> None:
        x = self._x().requires_grad_(True)
        out, _ = self.layer(x, self._state())
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.isfinite().all())

    def test_gate_receives_gradient(self) -> None:
        x = self._x().requires_grad_(True)
        out, _ = self.layer(x, self._state())
        out.sum().backward()
        self.assertIsNotNone(self.layer.gate.grad)

    def test_l2_output_mixed_by_gate(self) -> None:
        """
        At gate=0, sigmoid(gate)=0.5 → equal mix of L2 and SSM.
        Setting gate >> 0 should shift output toward L2 output.
        """
        x = self._x()
        state = self._state()
        with torch.no_grad():
            # Bias gate toward L2
            self.layer.gate.fill_(10.0)
            out_l2_biased, _ = self.layer(x, state)
            # Bias gate toward Mamba
            self.layer.gate.fill_(-10.0)
            out_ssm_biased, _ = self.layer(x, state)
        self.assertFalse(torch.allclose(out_l2_biased, out_ssm_biased))
        # Restore
        with torch.no_grad():
            self.layer.gate.fill_(0.0)

    def test_use_l2_flag_true(self) -> None:
        self.assertTrue(self.layer._use_l2)


# ---------------------------------------------------------------------------
# Tests for MambaLayer — without L2
# ---------------------------------------------------------------------------


class TestMambaLayerNoL2(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 32
        self.n_heads = 4
        self.d_k = self.d_model // self.n_heads
        self.layer = MambaLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_state=8,
            d_conv=3,
            expand=2,
            use_l2=False,
        )
        self.B, self.S = 2, 12

    def _state(self) -> MemoryState:
        return _zeros_state(self.B, self.n_heads, self.d_k, torch.device("cpu"))

    def _x(self) -> torch.Tensor:
        torch.manual_seed(11)
        return torch.randn(self.B, self.S, self.d_model)

    def test_no_l2_attribute(self) -> None:
        self.assertFalse(hasattr(self.layer, "l2"))
        self.assertFalse(hasattr(self.layer, "gate"))

    def test_use_l2_flag_false(self) -> None:
        self.assertFalse(self.layer._use_l2)

    def test_output_shape(self) -> None:
        out, _ = self.layer(self._x(), self._state())
        self.assertEqual(out.shape, (self.B, self.S, self.d_model))

    def test_state_passthrough(self) -> None:
        """State must be returned unchanged when L2 is disabled."""
        state = self._state()
        _, new_state = self.layer(self._x(), state)
        self.assertIs(new_state, state)

    def test_output_is_finite(self) -> None:
        out, _ = self.layer(self._x(), self._state())
        self.assertTrue(out.isfinite().all())

    def test_gradient_flows(self) -> None:
        x = self._x().requires_grad_(True)
        out, _ = self.layer(x, self._state())
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.isfinite().all())

    def test_output_equals_ssm_only(self) -> None:
        """No-L2 output must match calling ssm directly."""
        x = self._x()
        with torch.no_grad():
            out_layer, _ = self.layer(x, self._state())
            out_ssm = self.layer.ssm(x)
        self.assertTrue(torch.allclose(out_layer, out_ssm))


# ---------------------------------------------------------------------------
# Tests for DrexTransformer integration
# ---------------------------------------------------------------------------


class TestMambaTransformerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _small_config(use_mamba=True)
        self.model = DrexTransformer(self.config)
        self.B, self.S = 2, 16

    def _ids(self) -> torch.Tensor:
        return torch.randint(0, self.config.vocab_size, (self.B, self.S))

    def test_forward_shape(self) -> None:
        logits, states = self.model(self._ids())
        self.assertEqual(logits.shape, (self.B, self.S, self.config.vocab_size))
        self.assertEqual(len(states), self.config.n_layers)

    def test_output_finite(self) -> None:
        logits, _ = self.model(self._ids())
        self.assertTrue(logits.isfinite().all())

    def test_gradient_flows_to_embeddings(self) -> None:
        ids = self._ids()
        logits, _ = self.model(ids)
        logits.sum().backward()
        self.assertIsNotNone(self.model.token_emb.weight.grad)

    def test_layers_use_mamba_layer(self) -> None:
        for layer in self.model.layers:
            self.assertIsInstance(layer.attn, MambaLayer)

    def test_baseline_layers_use_hybrid_attention(self) -> None:
        from drex.models.attention import HybridAttention
        baseline_config = _small_config(use_mamba=False)
        baseline = DrexTransformer(baseline_config)
        for layer in baseline.layers:
            self.assertIsInstance(layer.attn, HybridAttention)

    def test_mamba_model_has_more_params_than_baseline(self) -> None:
        """
        Mamba has additional SSM parameters vs SlidingWindowAttention,
        so the Mamba model should have more total trainable params.
        """
        baseline = DrexTransformer(_small_config(use_mamba=False))
        mamba = DrexTransformer(_small_config(use_mamba=True))
        n_baseline = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        n_mamba = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
        self.assertGreater(n_mamba, n_baseline)

    def test_stateful_forward_accumulates_l2_memory(self) -> None:
        states = self.model.init_states(self.B, torch.device("cpu"))
        ids = self._ids()
        _, states2 = self.model(ids, states)
        # After one segment the L2 memory M should be non-zero
        initial_M = states[0].memory.M
        updated_M = states2[0].memory.M
        self.assertFalse(torch.allclose(initial_M, updated_M))

    def test_mamba_no_l2(self) -> None:
        """use_mamba=True with no_l2 disables InfiniAttention in all layers."""
        cfg = _small_config(use_mamba=True, use_l2=False)
        model = DrexTransformer(cfg)
        for layer in model.layers:
            self.assertIsInstance(layer.attn, MambaLayer)
            self.assertFalse(layer.attn._use_l2)

    def test_mamba_with_hdc_encoder(self) -> None:
        """Mamba and HDC encoder can be used together."""
        cfg = _small_config(
            use_mamba=True,
            use_hdc_encoder=True,
            hdc_dim=64,
            hdc_seed=1,
        )
        model = DrexTransformer(cfg)
        logits, _ = model(self._ids())
        self.assertEqual(logits.shape, (self.B, self.S, cfg.vocab_size))

    def test_checkpoint_roundtrip(self) -> None:
        """save_checkpoint + load_checkpoint must restore identical logits."""
        import tempfile
        from pathlib import Path

        from drex.utils.config import load_checkpoint, save_checkpoint

        ids = self._ids()
        with torch.no_grad():
            logits_before, _ = self.model(ids)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "mamba_test.safetensors"
            save_checkpoint(self.model, ckpt, step=0)
            load_checkpoint(self.model, ckpt)

        with torch.no_grad():
            logits_after, _ = self.model(ids)

        self.assertTrue(torch.allclose(logits_before, logits_after))


# ---------------------------------------------------------------------------
# Tests for DrexConfig Mamba fields
# ---------------------------------------------------------------------------


class TestDrexConfigMambaFields(unittest.TestCase):
    def test_default_use_mamba_false(self) -> None:
        cfg = DrexConfig()
        self.assertFalse(cfg.use_mamba)

    def test_default_mamba_d_state(self) -> None:
        cfg = DrexConfig()
        self.assertEqual(cfg.mamba_d_state, 16)

    def test_default_mamba_d_conv(self) -> None:
        cfg = DrexConfig()
        self.assertEqual(cfg.mamba_d_conv, 4)

    def test_default_mamba_expand(self) -> None:
        cfg = DrexConfig()
        self.assertEqual(cfg.mamba_expand, 2)

    def test_custom_mamba_fields(self) -> None:
        cfg = DrexConfig(use_mamba=True, mamba_d_state=32, mamba_d_conv=3, mamba_expand=4)
        self.assertTrue(cfg.use_mamba)
        self.assertEqual(cfg.mamba_d_state, 32)
        self.assertEqual(cfg.mamba_d_conv, 3)
        self.assertEqual(cfg.mamba_expand, 4)


# ---------------------------------------------------------------------------
# Tests for train.py CLI argument parsing
# ---------------------------------------------------------------------------


class TestTrainArgparseMamba(unittest.TestCase):
    def _parse(self, extra: list[str] | None = None) -> object:
        import sys
        from scripts.train import _parser

        argv = ["--steps", "1", "--max-chars", "500"] + (extra or [])
        p = _parser()
        return p.parse_args(argv)

    def test_use_mamba_default_false(self) -> None:
        args = self._parse()
        self.assertFalse(args.use_mamba)

    def test_use_mamba_flag(self) -> None:
        args = self._parse(["--use-mamba"])
        self.assertTrue(args.use_mamba)

    def test_mamba_d_state_default(self) -> None:
        args = self._parse()
        self.assertEqual(args.mamba_d_state, 16)

    def test_mamba_d_state_custom(self) -> None:
        args = self._parse(["--mamba-d-state", "32"])
        self.assertEqual(args.mamba_d_state, 32)

    def test_mamba_d_conv_default(self) -> None:
        args = self._parse()
        self.assertEqual(args.mamba_d_conv, 4)

    def test_mamba_d_conv_custom(self) -> None:
        args = self._parse(["--mamba-d-conv", "3"])
        self.assertEqual(args.mamba_d_conv, 3)

    def test_mamba_expand_default(self) -> None:
        args = self._parse()
        self.assertEqual(args.mamba_expand, 2)

    def test_mamba_expand_custom(self) -> None:
        args = self._parse(["--mamba-expand", "4"])
        self.assertEqual(args.mamba_expand, 4)


if __name__ == "__main__":
    unittest.main()
