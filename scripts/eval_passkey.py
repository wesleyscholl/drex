#!/usr/bin/env python3
"""
scripts/eval_passkey.py — Passkey recall sweep across context lengths.

Measures the fraction of trials where the model correctly recalls a 5-digit
passkey embedded in a long filler sequence. Sweeps from 2k to 32k tokens.

Usage (random-init baseline — expected ~0% accuracy):
    python scripts/eval_passkey.py

Evaluate a trained checkpoint:
    python scripts/eval_passkey.py --checkpoint checkpoints/step_0050000_final.safetensors

Custom sweep:
    python scripts/eval_passkey.py --lengths 2048 4096 8192 --trials 20

Report write rates alongside accuracy (requires --use-episodic-memory):
    python scripts/eval_passkey.py --report-write-rate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from drex.eval.passkey import PasskeyBenchmark
from drex.models.memory import MemoryModule, WRITE_RATE_LO, WRITE_RATE_HI
from drex.models.transformer import DrexConfig, DrexTransformer
from drex.utils.config import load_checkpoint


# Filler text used for density sweep prompts (same source as PasskeyBenchmark)
_DISTRACTOR_STR = "The quick brown fox jumps over the lazy dog. "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(args: argparse.Namespace, device: torch.device) -> DrexTransformer:
    """Build a DrexTransformer, optionally loading weights from a checkpoint."""
    config = DrexConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        vocab_size=256,
        window_size=args.window_size,
        max_seq_len=max(args.lengths) + args.window_size,
        dropout=0.0,
        use_l3=args.use_l3,
        l3_base_path=args.l3_path,
        use_episodic_memory=args.use_episodic_memory,
        episodic_gate_thresh=args.episodic_gate_thresh,
    )
    model = DrexTransformer(config).to(device)

    if args.checkpoint:
        step = load_checkpoint(model, args.checkpoint)
        print(f"Loaded checkpoint from '{args.checkpoint}' (step {step})", flush=True)
    else:
        print("No checkpoint supplied — evaluating randomly-initialised model.", flush=True)

    model.eval()
    return model


def _collect_write_rates(model: DrexTransformer) -> list[float]:
    """Return last_write_rate() from every MemoryModule in the model."""
    return [m.last_write_rate() for m in model.modules() if isinstance(m, MemoryModule)]


def _report_write_rates(
    model: DrexTransformer,
    lengths: list[int],
    device: torch.device,
    vocab_size: int = 256,
) -> None:
    """
    Run a single forward pass per context length and print a write-rate table.

    Each pass uses a randomly-generated token sequence of the target length so
    that the model's MemoryModule.last_write_rate() reflects that exact length.
    """
    print("\nWrite-rate sweep (MemoryModule OR-gate firing fraction per layer):")
    header = f"{'Context':>10}" + "  mean_wr  min_wr  max_wr"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    with torch.no_grad():
        for length in sorted(lengths):
            ids = torch.randint(0, vocab_size, (1, length), device=device)
            model(ids)
            rates = _collect_write_rates(model)
            if not rates:
                print(f"  {length:>8,}  (no MemoryModule instances found)")
                continue
            mean_wr = sum(rates) / len(rates)
            print(
                f"  {length:>8,}  {mean_wr:>7.3f}  {min(rates):>6.3f}  {max(rates):>6.3f}"
            )
    print(sep)
    print()


def _make_density_prompt(
    context_len: int,
    rho: float,
    seed: int,
    vocab_size: int = 256,
) -> list[int]:
    """
    Build a token list with approximately rho * context_len tokens occupied by
    key sentences, distributed evenly throughout the sequence.

    Used only for write-rate density measurement; not an accuracy benchmark.
    rho=0.08 = low density; rho=0.30 = high density.
    """
    import random as _rng_module

    rng = _rng_module.Random(seed)
    template = "The number is: {n}. "
    avg_len = len(template.format(n="12345"))
    n_keys = max(1, round(rho * context_len / avg_len))
    key_seqs = [
        [min(ord(c), vocab_size - 1) for c in template.format(n=rng.randint(10000, 99999))]
        for _ in range(n_keys)
    ]
    filler_unit = [min(ord(c), vocab_size - 1) for c in _DISTRACTOR_STR]
    key_total = sum(len(s) for s in key_seqs)
    gap = max(0, (context_len - key_total) // (n_keys + 1))
    tokens: list[int] = []
    for ks in key_seqs:
        fill_needed = gap
        while fill_needed > 0:
            chunk = filler_unit[:fill_needed]
            tokens.extend(chunk)
            fill_needed -= len(chunk)
            if len(chunk) == 0:
                break
        tokens.extend(ks)
    while len(tokens) < context_len:
        tokens.extend(filler_unit)
    return [min(t, vocab_size - 1) for t in tokens[:context_len]]


def _report_density_sweep(
    model: DrexTransformer,
    lengths: list[int],
    densities: list[float],
    device: torch.device,
    n_trials: int,
    vocab_size: int = 256,
) -> None:
    """
    Print a write-rate table swept over (context_length, key_density) cells.

    Verifies that the OR-gate threshold produces write rates within
    [WRITE_RATE_LO, WRITE_RATE_HI] across the supported density range.
    """
    print("\nWrite-rate density sweep (mean wr per layer across trials):")
    header = f"  {'Length':>8}  {'Density':>8}  mean_wr  min_wr  max_wr"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    with torch.no_grad():
        for length in sorted(lengths):
            for rho in sorted(densities):
                all_rates: list[float] = []
                for s in range(n_trials):
                    ids_list = _make_density_prompt(length, rho, s, vocab_size)
                    ids = torch.tensor([ids_list], dtype=torch.long, device=device)
                    model(ids)
                    all_rates.extend(_collect_write_rates(model))
                if not all_rates:
                    print(f"  {length:>8,}  {rho:>8.2f}  (no MemoryModule found)")
                    continue
                mean_wr = sum(all_rates) / len(all_rates)
                flag = ""
                if mean_wr < WRITE_RATE_LO or mean_wr > WRITE_RATE_HI:
                    flag = "  [WARNING: outside valid range]"
                print(
                    f"  {length:>8,}  {rho:>8.2f}"
                    f"  {mean_wr:>7.3f}  {min(all_rates):>6.3f}  {max(all_rates):>6.3f}"
                    + flag
                )
    print(sep)
    print()


def _print_table(results: dict[int, dict[int, float]]) -> None:
    lengths = sorted(next(iter(results.values())).keys())
    header = f"{'Config':<30}" + "".join(f"  {l//1024:>4}k" for l in lengths)
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for label, row in results.items():
        vals = "".join(f"  {row[l]:>5.1%}" for l in lengths)
        print(f"{str(label):<30}{vals}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> None:
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    model = _make_model(args, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)
    print(f"Context lengths: {args.lengths}", flush=True)
    print(f"Trials per length: {args.trials}", flush=True)
    print()

    bench = PasskeyBenchmark(
        model=model,
        context_lengths=args.lengths,
        n_trials=args.trials,
        device=str(device),
        segment_len=args.window_size,
    )

    print("Running passkey recall sweep …", flush=True)
    results = bench.run()

    # Display
    label = Path(args.checkpoint).stem if args.checkpoint else "random-init"
    lengths = sorted(results)
    header = f"{'model':<40}" + "".join(f"  {l//1024:>4}k" for l in lengths)
    sep = "-" * len(header)
    vals = "".join(f"  {results[l]:>5.1%}" for l in lengths)

    print(sep)
    print(header)
    print(sep)
    print(f"{label:<40}{vals}")
    print(sep)

    # Summary
    avg = sum(results.values()) / len(results)
    best_len = max(results, key=results.get)
    print(f"\nAverage accuracy : {avg:.1%}")
    print(f"Best accuracy    : {results[best_len]:.1%}  (context length {best_len:,})")

    if avg < 0.05:
        print("\n[note] Accuracy is near chance — the model has not been trained yet.")
        print("       Run scripts/train.py first, then re-evaluate with --checkpoint.")

    # Optional write-rate report
    if args.report_write_rate:
        if not args.use_episodic_memory:
            print(
                "\n[note] --report-write-rate has no effect without --use-episodic-memory."
            )
        else:
            _report_write_rates(model, args.lengths, device)

    # Optional density sweep
    if args.density:
        if not args.use_episodic_memory:
            print("\n[note] --density has no effect without --use-episodic-memory.")
        else:
            _report_density_sweep(
                model,
                args.lengths,
                args.density,
                device,
                n_trials=args.density_trials,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Passkey recall sweep across context lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation
    p.add_argument(
        "--lengths", type=int, nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
        metavar="N",
        help="Context lengths to evaluate (space-separated token counts)",
    )
    p.add_argument("--trials", type=int, default=10,
                   help="Number of independent passkey trials per context length")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a .safetensors checkpoint (omit for random-init baseline)")

    # Model architecture (must match checkpoint if loading one)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--window-size", type=int, default=512,
                   help="Sliding-window size (must match training segment length)")

    # L3
    p.add_argument("--use-l3", action="store_true")
    p.add_argument("--l3-path", type=str, default="/tmp/drex_l3")

    # Episodic memory (Phase 13)
    p.add_argument("--use-episodic-memory", action="store_true",
                   help="Enable MemoryModule per layer (Phase 13 validated architecture)")
    p.add_argument("--episodic-gate-thresh", type=float, default=0.70,
                   help="OR-gate threshold for MemoryModule (thresh*=0.70 per exp_48_1)")
    p.add_argument("--report-write-rate", action="store_true",
                   help="Print MemoryModule write-rate table after accuracy sweep")
    p.add_argument(
        "--density",
        type=float,
        nargs="*",
        default=[],
        metavar="RHO",
        help=(
            "Key-sentence density values for write-rate density sweep "
            "(rho = fraction of context occupied by key sentences). "
            "Requires --use-episodic-memory. Example: --density 0.08 0.30"
        ),
    )
    p.add_argument(
        "--density-trials",
        type=int,
        default=5,
        help="Number of random seeds per (length, density) cell in the density sweep",
    )

    # Infrastructure
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])

    return p


if __name__ == "__main__":
    run_eval(_parser().parse_args())
