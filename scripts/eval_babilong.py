#!/usr/bin/env python3
"""
scripts/eval_babilong.py — BABILong-style Q&A sweep across context lengths.

Measures accuracy on 5 synthetic reasoning tasks embedded in a long filler
sequence. Tasks range from single-fact retrieval (Task 1) to count-after-drop
reasoning (Task 5). Sweeps across user-specified context lengths.

Usage (random-init baseline — expected ~0% accuracy):
    python scripts/eval_babilong.py

Evaluate a trained checkpoint:
    python scripts/eval_babilong.py --checkpoint checkpoints/step_0050000_final.safetensors

Custom sweep:
    python scripts/eval_babilong.py --lengths 2048 4096 8192 --trials 20 --tasks 1 2

With episodic memory and write-rate reporting:
    python scripts/eval_babilong.py \\
        --use-episodic-memory --report-write-rate \\
        --checkpoint checkpoints/step_0050000_final.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from drex.eval.babilong import BABILongBenchmark
from drex.models.memory import MemoryModule
from drex.models.transformer import DrexConfig, DrexTransformer
from drex.utils.config import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NAMES: dict[int, str] = {
    1: "Task 1 (single fact)",
    2: "Task 2 (two facts)",
    3: "Task 3 (three facts)",
    4: "Task 4 (possession)",
    5: "Task 5 (count after drop)",
}


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


def _print_results_table(
    results: dict[int, dict[int, float]],
    lengths: list[int],
    tasks: list[int],
) -> None:
    """Print a markdown-style table with tasks as rows and context lengths as columns."""
    sorted_lengths = sorted(lengths)
    name_w = max(len(TASK_NAMES.get(t, f"Task {t}")) for t in tasks) + 2
    header_cols = "".join(f"  {l // 1024:>4}k" for l in sorted_lengths)
    header = f"{'Task':<{name_w}}{header_cols}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    per_length_totals: dict[int, float] = {l: 0.0 for l in sorted_lengths}

    for task_id in tasks:
        row = results.get(task_id, {})
        task_name = TASK_NAMES.get(task_id, f"Task {task_id}")
        vals = "".join(f"  {row.get(l, 0.0):>5.1%}" for l in sorted_lengths)
        print(f"{task_name:<{name_w}}{vals}")
        for l in sorted_lengths:
            per_length_totals[l] += row.get(l, 0.0)

    print(sep)
    avg_vals = "".join(
        f"  {per_length_totals[l] / len(tasks):>5.1%}" for l in sorted_lengths
    )
    print(f"{'Average':<{name_w}}{avg_vals}")
    print(sep)


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
    print(f"Tasks: {args.tasks}", flush=True)
    print(f"Context lengths: {args.lengths}", flush=True)
    print(f"Trials per cell: {args.trials}", flush=True)
    print()

    bench = BABILongBenchmark(
        model=model,
        context_lengths=args.lengths,
        tasks=tuple(args.tasks),
        n_trials=args.trials,
        device=device,
        segment_len=args.window_size,
    )

    print("Running BABILong evaluation …", flush=True)
    results = bench.run()
    print()

    _print_results_table(results, args.lengths, args.tasks)

    # Summary
    all_accs = [
        results[t][l]
        for t in args.tasks
        for l in args.lengths
        if t in results and l in results[t]
    ]
    grand_mean = sum(all_accs) / len(all_accs) if all_accs else 0.0
    print(f"\nGrand mean accuracy : {grand_mean:.1%}")

    if args.tasks and args.lengths:
        best_task = max(
            args.tasks,
            key=lambda t: sum(results.get(t, {}).get(l, 0.0) for l in args.lengths),
        )
        best_length = max(
            args.lengths,
            key=lambda l: sum(results.get(t, {}).get(l, 0.0) for t in args.tasks),
        )
        task_mean = sum(results.get(best_task, {}).get(l, 0.0) for l in args.lengths) / len(
            args.lengths
        )
        len_mean = sum(results.get(t, {}).get(best_length, 0.0) for t in args.tasks) / len(
            args.tasks
        )
        print(
            f"Best task         : {TASK_NAMES.get(best_task, f'Task {best_task}')} — "
            f"mean {task_mean:.1%}"
        )
        print(
            f"Best length       : {best_length:,} — mean {len_mean:.1%}"
        )

    if grand_mean < 0.167:
        print("\n[note] Accuracy is near chance — the model has not been trained.")
        print("       Run scripts/train.py first, then re-evaluate with --checkpoint.")

    # Optional write-rate report
    if args.report_write_rate:
        if not args.use_episodic_memory:
            print(
                "\n[note] --report-write-rate has no effect without --use-episodic-memory."
            )
        else:
            _report_write_rates(model, args.lengths, device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BABILong Q&A accuracy sweep across context lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Evaluation
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        metavar="N",
        help="Context lengths to evaluate (space-separated token counts)",
    )
    p.add_argument("--trials", type=int, default=10,
                   help="Number of independent trials per (task, context_length) cell")
    p.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        metavar="T",
        help="Task IDs to evaluate: 1=single fact, 2=two facts, 3=three facts, "
             "4=possession, 5=count after drop",
    )
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

    # Episodic memory (Phase 13 validated architecture)
    p.add_argument("--use-episodic-memory", action="store_true",
                   help="Enable MemoryModule per layer (Phase 13 validated architecture)")
    p.add_argument("--episodic-gate-thresh", type=float, default=0.70,
                   help="OR-gate threshold for MemoryModule (thresh*=0.70 per exp_48_1)")
    p.add_argument("--report-write-rate", action="store_true",
                   help="Print MemoryModule write-rate table after accuracy sweep")

    # Infrastructure
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])

    return p


if __name__ == "__main__":
    run_eval(_parser().parse_args())
