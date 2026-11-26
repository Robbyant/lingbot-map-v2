# Copyright 2026 The Robbyant Team Authors. All rights reserved.

"""Run one pure Online prediction per selected scene; no timing repetitions."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "benchmark"

import argparse
from .benchmark.core.storage import source_identity
from .methods.lingbot_map2 import load_model, run_scene
from .prepare import load_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--output")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--device")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", help="base YAML under benchmark/configs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved settings without data access or CUDA",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="with --resume, archive incomplete attempts and restart only those scenes",
    )
    args = parser.parse_args()
    if args.config:
        if (
            args.plan is not None
            or args.checkpoint is not None
            or args.checkpoint_config is not None
            or args.output is not None
            or args.device is not None
            or args.compile is not None
            or args.depth is not None
        ):
            parser.error("do not mix --config with low-level input/runtime flags")
        from .benchmark.core.config import configured_stage

        try:
            configured_stage(
                args.config,
                "run",
                scenes=args.scene,
                resume=args.resume,
                dry_run=args.dry_run,
                retry_failed=args.retry_failed,
            )
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as error:
            parser.exit(2, f"{error}\nSee benchmark/README.md#data-preparation.\n")
        return

    if args.dry_run or args.resume or args.retry_failed:
        parser.error("these pipeline flags require --config")
    args.compile = True if args.compile is None else args.compile
    args.depth = True if args.depth is None else args.depth
    args.device = args.device or "cuda:0"
    if not all([args.plan, args.checkpoint, args.output]):
        parser.error("provide --config, or --plan --checkpoint --output")
    plan = load_plan(args.plan)
    if plan["protocol"]["kind"] == "dense" and not args.depth:
        parser.error("dense reconstruction requires --depth")
    rows = plan["scenes"]
    if args.scene:
        if set(args.scene) - {r["name"] for r in rows}:
            parser.error("requested scene is absent from the plan")
        rows = [r for r in rows if r["name"] in args.scene]
    output = Path(args.output)
    for row in rows:
        if (output / row["name"]).exists():
            raise FileExistsError(
                "existing predictions are immutable; select unfinished scenes or a new output root"
            )
    model = load_model(
        plan,
        args.checkpoint,
        config=args.checkpoint_config,
        depth=args.depth,
        device=args.device,
        compile=args.compile,
    )
    code_id = source_identity()
    for row in rows:
        run_scene(
            model, row, plan["protocol"], output / row["name"], plan["plan_id"], code_id
        )
        print(f"Completed {row['name']}", flush=True)


if __name__ == "__main__":
    main()
