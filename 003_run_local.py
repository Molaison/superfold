#!/usr/bin/env python3
"""
Run Slurm array scripts locally (simulate SLURM env vars).

Copied/adapted from `bioinformatics/alphafold/003_run_local.py` so the SuperFold
repo has an in-tree local runner.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_available_gpus() -> List[int]:
    """Get available GPU indices via nvidia-smi (best-effort)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = [
            int(line.strip())
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: could not detect GPUs; assuming a single GPU (id=0).")
        return [0]


def read_file_list(file_list_path: Path) -> List[str]:
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")
    with open(file_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def distribute_files(files: List[str], num_tasks: int) -> List[List[str]]:
    task_files = [[] for _ in range(num_tasks)]
    for idx, file in enumerate(files):
        task_files[idx % num_tasks].append(file)
    return task_files


def run_task(
    script_path: Path,
    task_id: int,
    total_tasks: int,
    gpu_id: Optional[int] = None,
    env_vars: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()

    env["SLURM_ARRAY_TASK_ID"] = str(task_id)
    env["SLURM_ARRAY_TASK_COUNT"] = str(total_tasks)
    env["SLURM_ARRAY_TASK_MIN"] = "1"
    env["SLURM_ARRAY_TASK_MAX"] = str(total_tasks)

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if env_vars:
        env.update(env_vars)

    print(
        f"[task {task_id}/{total_tasks}] start (GPU: {gpu_id if gpu_id is not None else 'CPU'})"
    )
    result = subprocess.run(["bash", str(script_path)], env=env)
    if result.returncode == 0:
        print(f"[task {task_id}/{total_tasks}] done ✓")
    else:
        print(f"[task {task_id}/{total_tasks}] failed ✗ (code={result.returncode})")
    return result


def run_parallel(
    script_path: Path, num_tasks: int, gpus: List[int], env_vars: Optional[dict] = None
) -> int:
    import concurrent.futures

    task_gpus: List[Optional[int]] = []
    for i in range(num_tasks):
        task_gpus.append(gpus[i % len(gpus)] if gpus else None)

    print(f"Parallel: {num_tasks} tasks on {len(gpus)} GPU(s)")
    print(f"GPU assignment: {task_gpus}")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(gpus))) as executor:
        futures = []
        for task_id in range(1, num_tasks + 1):
            futures.append(
                executor.submit(
                    run_task, script_path, task_id, num_tasks, task_gpus[task_id - 1], env_vars
                )
            )
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    success = sum(1 for r in results if r.returncode == 0)
    failed = len(results) - success
    print("=" * 60)
    print(f"Done: {success}/{num_tasks} tasks succeeded")
    if failed:
        print(f"Failed: {failed} task(s)")
        return 1
    return 0


def run_sequential(
    script_path: Path,
    num_tasks: int,
    gpu_id: Optional[int] = None,
    env_vars: Optional[dict] = None,
) -> int:
    print(f"Sequential: {num_tasks} tasks")
    if gpu_id is not None:
        print(f"Using GPU: {gpu_id}")
    print("=" * 60)
    failed: List[int] = []
    for task_id in range(1, num_tasks + 1):
        r = run_task(script_path, task_id, num_tasks, gpu_id, env_vars)
        if r.returncode != 0:
            failed.append(task_id)
    print("=" * 60)
    if failed:
        print(f"Failed tasks: {failed}")
        return 1
    print("All tasks completed ✓")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Slurm array bash script locally by simulating SLURM_* env vars."
    )
    parser.add_argument("script", type=Path, help="Path to the Slurm array script (bash).")
    parser.add_argument("--num-tasks", "-n", type=int, required=True, help="Number of tasks (array length).")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids (e.g. 0,1). Default: auto-detect.")
    parser.add_argument("--no-gpu", action="store_true", help="Do not set CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially (no parallelism).")
    parser.add_argument("--env", type=str, action="append", help="Extra env var (KEY=VALUE). Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Preview assignment; do not execute.")
    args = parser.parse_args()

    if not args.script.exists():
        print(f"Error: script not found: {args.script}", file=sys.stderr)
        return 2

    if args.no_gpu:
        gpus: List[int] = []
    elif args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus = get_available_gpus()

    env_vars = {}
    if args.env:
        for item in args.env:
            if "=" not in item:
                print(f"Warning: ignoring invalid env format: {item}", file=sys.stderr)
                continue
            k, v = item.split("=", 1)
            env_vars[k] = v

    print("=" * 60)
    print("Local run configuration")
    print("=" * 60)
    print(f"script: {args.script}")
    print(f"tasks:  {args.num_tasks}")
    print(f"gpus:   {gpus if gpus else 'CPU only'}")
    print(f"mode:   {'sequential' if args.sequential else 'parallel'}")
    if env_vars:
        print(f"env:    {env_vars}")
    print("=" * 60)

    if args.dry_run:
        script_dir = args.script.parent.resolve()
        run_dir = script_dir.parent
        file_list = run_dir / "inputs" / "file_list.txt"
        if not file_list.exists():
            print("No inputs/file_list.txt found; nothing to preview.")
            return 0
        files = read_file_list(file_list)
        print(f"file_list: {file_list} ({len(files)} items)")
        task_files = distribute_files(files, args.num_tasks)
        for i, items in enumerate(task_files, start=1):
            gpu_info = f"GPU {gpus[(i - 1) % len(gpus)]}" if gpus else "CPU"
            print(f"  task {i} ({gpu_info}): {len(items)} file(s)")
            if items:
                print(f"    first: {items[0]}")
        return 0

    if args.sequential or len(gpus) <= 1:
        gpu_id = gpus[0] if gpus else None
        return run_sequential(args.script, args.num_tasks, gpu_id, env_vars)
    return run_parallel(args.script, args.num_tasks, gpus, env_vars)


if __name__ == "__main__":
    raise SystemExit(main())
