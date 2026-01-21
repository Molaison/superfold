#!/usr/bin/env python3
"""
SuperFold run directory preparer (user-friendly wrapper).

Modeled after `001_prepare_af3_input.py` in this repo:
- Create a numbered run directory
- Normalize/split FASTA inputs into fewer chunk files
- Generate a file list for dynamic Slurm array distribution
- Generate Slurm/local run scripts

This script does NOT run SuperFold. It only prepares a runnable folder.

Typical:
  python 001_prepare_superfold_run.py -i inputs.fasta --weights-dir /path/to/alphafold_weights
  cd 001_superfold_run/scripts
  sbatch --array=1-4 run_prediction.sh

Local dry-run (no SuperFold deps needed):
  SUPERFOLD_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 SLURM_ARRAY_TASK_COUNT=4 bash run_prediction.sh
"""

import argparse
import gzip
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SUPPORTED_SUFFIXES = (
    ".fa",
    ".fasta",
    ".fa.gz",
    ".fasta.gz",
    ".pdb",
    ".pdb.gz",
    ".silent",
    ".silent.gz",
)


def _is_supported(path: Path) -> bool:
    s = path.name.lower()
    return any(s.endswith(suf) for suf in SUPPORTED_SUFFIXES)


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def sanitize_name(name: str) -> str:
    # Keep it file-name friendly: allow alnum, dot, dash, underscore.
    name = name.strip().replace(":", "/")
    name = name.replace("\\", "/")
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "target"


@dataclass(frozen=True)
class FastaRecord:
    original_name: str
    safe_name: str
    sequence: str
    source: str


def read_fasta_records(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    name: Optional[str] = None
    seq_parts: List[str] = []

    with _open_text(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(seq_parts).strip()
                    if seq:
                        # SuperFold supports chain breaks as "/" or ":"; normalize ":" -> "/".
                        records.append((name, seq.replace(":", "/")))
                name = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

    if name is not None:
        seq = "".join(seq_parts).strip()
        if seq:
            records.append((name, seq.replace(":", "/")))

    return records


def collect_input_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    it: Iterable[Path] = input_path.rglob("*") if recursive else input_path.glob("*")
    files = [p for p in it if p.is_file() and _is_supported(p)]
    return sorted(files)


def chunk_records(records: Sequence[FastaRecord], chunk_size: int) -> List[List[FastaRecord]]:
    if chunk_size <= 0:
        raise ValueError(f"--fasta-chunk-size must be > 0, got {chunk_size}")
    chunks: List[List[FastaRecord]] = []
    current: List[FastaRecord] = []
    for r in records:
        current.append(r)
        if len(current) >= chunk_size:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def write_fasta_chunk(path: Path, chunk: Sequence[FastaRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in chunk:
            f.write(f">{rec.safe_name}\n")
            # Keep original "/" chain breaks if any.
            seq = rec.sequence.replace(" ", "").strip()
            # Wrap at 80 chars for readability.
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def create_run_directory(base_dir: Path, run_number: Optional[int]) -> Path:
    if run_number is not None:
        run_dir = base_dir / f"{run_number:03d}_superfold_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    existing = sorted(base_dir.glob("*_superfold_run"))
    next_num = 1
    if existing:
        # Best-effort: extract numeric prefix like 001_superfold_run.
        for p in reversed(existing):
            prefix = p.name.split("_", 1)[0]
            if prefix.isdigit():
                next_num = int(prefix) + 1
                break
        else:
            next_num = len(existing) + 1
    run_dir = base_dir / f"{next_num:03d}_superfold_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_weights_pointer(superfold_root: Path, weights_dir: Optional[Path]) -> Optional[Path]:
    """
    SuperFold reads weights from `<superfold_root>/alphafold_weights.pth`.
    If `weights_dir` is provided, write/update that file to point at it.
    """
    weights_pth = superfold_root / "alphafold_weights.pth"
    if weights_dir is None:
        return weights_pth if weights_pth.exists() else None

    weights_dir = weights_dir.expanduser().resolve()
    if not weights_dir.exists():
        raise FileNotFoundError(f"--weights-dir not found: {weights_dir}")
    if not (weights_dir / "params").exists():
        raise FileNotFoundError(
            f"--weights-dir must contain a child directory named 'params/': {weights_dir}"
        )

    weights_pth.write_text(str(weights_dir) + "\n", encoding="utf-8")
    return weights_pth


def render_run_prediction_sh(
    *,
    run_dir: Path,
    superfold_root: Path,
    python_bin: str,
    num_parts_default: int,
    cpus_per_task: int,
    extra_args: List[str],
) -> str:
    input_dir = (run_dir / "inputs").absolute()
    output_dir = (run_dir / "outputs").absolute()
    log_dir = (run_dir / "logs").absolute()
    file_list = (input_dir / "file_list.txt").absolute()
    run_superfold = (superfold_root / "run_superfold.py").absolute()

    # Extra args must be shell-escaped.
    extra = " ".join(shlex.quote(x) for x in extra_args)

    return f"""#!/usr/bin/env bash
#SBATCH --job-name=superfold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.out
#SBATCH --array=1-{num_parts_default}

set -euo pipefail

RUN_DIR={shlex.quote(str(run_dir.absolute()))}
INPUT_DIR={shlex.quote(str(input_dir))}
OUT_DIR={shlex.quote(str(output_dir))}
LOG_DIR={shlex.quote(str(log_dir))}
FILE_LIST={shlex.quote(str(file_list))}

PYTHON={shlex.quote(python_bin)}
RUN_SUPERFOLD={shlex.quote(str(run_superfold))}

mkdir -p "$OUT_DIR" "$LOG_DIR"

mapfile -t ALL_FILES < "$FILE_LIST"
TOTAL_FILES=${{#ALL_FILES[@]}}

TOTAL_TASKS=${{SLURM_ARRAY_TASK_COUNT:-{num_parts_default}}}
TASK_ID=${{SLURM_ARRAY_TASK_ID:-1}}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$TOTAL_TASKS" ]; then
  echo "[error] Invalid SLURM_ARRAY_TASK_ID=$TASK_ID (TOTAL_TASKS=$TOTAL_TASKS)" >&2
  exit 2
fi

ARGS=()
for ((i=TASK_ID-1; i<TOTAL_FILES; i+=TOTAL_TASKS)); do
  rel="${{ALL_FILES[$i]}}"
  ARGS+=("$INPUT_DIR/$rel")
done

echo "[task] task_id=$TASK_ID total_tasks=$TOTAL_TASKS total_files=$TOTAL_FILES assigned_files=${{#ARGS[@]}}"
if [ "${{#ARGS[@]}}" -gt 0 ]; then
  for f in "${{ARGS[@]}}"; do
    echo "[task] file=$f"
  done
fi

if [ "${{#ARGS[@]}}" -eq 0 ]; then
  echo "[task] nothing to do"
  exit 0
fi

if [ "${{SUPERFOLD_DRY_RUN:-0}}" = "1" ]; then
  echo "[dry-run] $PYTHON $RUN_SUPERFOLD (inputs...=${{#ARGS[@]}}) --out_dir $OUT_DIR {extra}"
  exit 0
fi

exec "$PYTHON" "$RUN_SUPERFOLD" "${{ARGS[@]}}" --out_dir "$OUT_DIR" {extra}
"""


def render_local_wrapper_sh(run_dir: Path, num_parts_default: int) -> str:
    run_prediction = (run_dir / "scripts" / "run_prediction.sh").absolute()
    return f"""#!/usr/bin/env bash
set -euo pipefail

NUM_TASKS="${{1:-{num_parts_default}}}"
SCRIPT={shlex.quote(str(run_prediction))}

for ((tid=1; tid<=NUM_TASKS; tid++)); do
  echo "[local] running task $tid/$NUM_TASKS"
  SLURM_ARRAY_TASK_ID="$tid" SLURM_ARRAY_TASK_COUNT="$NUM_TASKS" bash "$SCRIPT"
done
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a SuperFold run directory (inputs/scripts/outputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input", type=Path, required=True, help="Input FASTA/PDB/silent file or directory.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Run directory to create (default: auto-numbered *_superfold_run).")
    parser.add_argument("--run", type=int, default=None, help="Explicit run number (used only when --output is not set).")
    parser.add_argument("--recursive", action="store_true", help="When input is a directory, search recursively.")

    parser.add_argument(
        "--superfold-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to the superfold repo (where run_superfold.py lives).",
    )
    parser.add_argument("--python", dest="python_bin", default="python3", help="Python interpreter used in generated scripts.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Path to AlphaFold weights directory (must contain params/). Writes superfold/alphafold_weights.pth.",
    )

    parser.add_argument("--num-parts", type=int, default=4, help="Default parallelism (Slurm array length). You can override at submit time.")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task in generated Slurm script.")
    parser.add_argument(
        "--fasta-chunk-size",
        type=int,
        default=32,
        help="Split FASTA records into chunk files with up to this many records each (reduces small-file explosion).",
    )

    # SuperFold passthroughs (a curated set). We embed them into run scripts.
    parser.add_argument("--mock-msa-depth", type=int, default=1, help="SuperFold: --mock_msa_depth")
    parser.add_argument("--models", nargs="+", default=["all"], help="SuperFold: --models (default: all)")
    parser.add_argument("--nstruct", type=int, default=1, help="SuperFold: --nstruct")
    parser.add_argument("--seed-start", type=int, default=0, help="SuperFold: --seed_start")
    parser.add_argument("--num-ensemble", type=int, default=1, help="SuperFold: --num_ensemble")
    parser.add_argument("--max-recycles", type=int, default=3, help="SuperFold: --max_recycles")
    parser.add_argument("--recycle-tol", type=float, default=0.0, help="SuperFold: --recycle_tol")
    parser.add_argument("--output-pae", action="store_true", help="SuperFold: --output_pae")
    parser.add_argument("--output-summary", action="store_true", help="SuperFold: --output_summary")
    parser.add_argument("--enable-dropout", action="store_true", help="SuperFold: --enable_dropout")
    parser.add_argument("--pct-seq-mask", type=float, default=0.15, help="SuperFold: --pct_seq_mask")
    parser.add_argument("--overwrite", action="store_true", help="SuperFold: --overwrite (otherwise checkpoints by default)")
    parser.add_argument("--initial-guess", nargs="?", const=True, default=False, help="SuperFold: --initial_guess [PATH|true]")
    parser.add_argument("--reference-pdb", type=str, default=None, help="SuperFold: --reference_pdb PATH")

    args = parser.parse_args()

    superfold_root = args.superfold_root.expanduser().resolve()
    run_superfold = superfold_root / "run_superfold.py"
    if not run_superfold.exists():
        raise FileNotFoundError(f"Could not find run_superfold.py under --superfold-root: {run_superfold}")

    # Create run directory layout.
    if args.output is not None:
        run_dir = args.output.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(Path.cwd(), args.run)

    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    scripts_dir = run_dir / "scripts"
    logs_dir = run_dir / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    weights_pointer = ensure_weights_pointer(superfold_root, args.weights_dir)

    # Collect inputs and normalize FASTA into chunks inside run_dir.
    input_files = collect_input_files(args.input.expanduser().resolve(), args.recursive)
    if not input_files:
        raise SystemExit(f"No supported input files found under: {args.input}")

    manifest = {
        "superfold_root": str(superfold_root),
        "run_dir": str(run_dir),
        "weights_pointer": str(weights_pointer) if weights_pointer else None,
        "input": str(args.input),
        "generated_at_cwd": str(Path.cwd()),
        "fasta_chunk_size": args.fasta_chunk_size,
        "notes": "FASTA headers are sanitized into safe names used for SuperFold output prefixes.",
        "records": [],
        "non_fasta_inputs": [],
    }

    # Create stable, safe paths under inputs/.
    rel_paths: List[str] = []

    # 1) Chunk all FASTA records into inputs/chunks/*.fasta
    fasta_records: List[FastaRecord] = []
    seen_names: dict = {}
    for p in input_files:
        lower = p.name.lower()
        if lower.endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz")):
            for original_name, seq in read_fasta_records(p):
                base = sanitize_name(original_name)
                count = seen_names.get(base, 0) + 1
                seen_names[base] = count
                safe = base if count == 1 else f"{base}_{count}"
                fasta_records.append(
                    FastaRecord(
                        original_name=original_name,
                        safe_name=safe,
                        sequence=seq,
                        source=str(p),
                    )
                )
                manifest["records"].append(
                    {
                        "original_name": original_name,
                        "safe_name": safe,
                        "source": str(p),
                        "length": len(seq.replace("/", "")),
                    }
                )

    if fasta_records:
        chunks = chunk_records(fasta_records, args.fasta_chunk_size)
        chunks_dir = inputs_dir / "chunks"
        for idx, chunk in enumerate(chunks, start=1):
            chunk_path = chunks_dir / f"chunk_{idx:04d}.fasta"
            write_fasta_chunk(chunk_path, chunk)
            rel_paths.append(str(chunk_path.relative_to(inputs_dir)))

    # 2) For non-FASTA inputs (pdb/silent), create symlinks under inputs/files/
    files_dir = inputs_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    non_fasta = [p for p in input_files if not p.name.lower().endswith((".fa", ".fasta", ".fa.gz", ".fasta.gz"))]
    for idx, p in enumerate(non_fasta, start=1):
        base = sanitize_name(p.name)
        link_name = f"{idx:04d}_{base}"
        link_path = files_dir / link_name
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(p)
        except OSError:
            # Fallback: copy if symlinks not supported.
            link_path.write_bytes(p.read_bytes())
        rel_paths.append(str(link_path.relative_to(inputs_dir)))
        manifest["non_fasta_inputs"].append({"source": str(p), "linked_as": str(link_path)})

    rel_paths = sorted(rel_paths)
    file_list_path = inputs_dir / "file_list.txt"
    file_list_path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")

    (inputs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build embedded SuperFold args.
    extra_args: List[str] = []
    extra_args += ["--mock_msa_depth", str(args.mock_msa_depth)]
    extra_args += ["--models", *args.models]
    extra_args += ["--nstruct", str(args.nstruct)]
    extra_args += ["--seed_start", str(args.seed_start)]
    extra_args += ["--num_ensemble", str(args.num_ensemble)]
    extra_args += ["--max_recycles", str(args.max_recycles)]
    extra_args += ["--recycle_tol", str(args.recycle_tol)]
    extra_args += ["--pct_seq_mask", str(args.pct_seq_mask)]
    if args.output_pae:
        extra_args.append("--output_pae")
    if args.output_summary:
        extra_args.append("--output_summary")
    if args.enable_dropout:
        extra_args.append("--enable_dropout")
    if args.overwrite:
        extra_args.append("--overwrite")
    if args.initial_guess is not False:
        # Either True (flag only) or a path.
        extra_args += ["--initial_guess"]
        if args.initial_guess is not True:
            extra_args.append(str(args.initial_guess))
    if args.reference_pdb:
        extra_args += ["--reference_pdb", str(args.reference_pdb)]

    run_prediction_sh = scripts_dir / "run_prediction.sh"
    run_prediction_sh.write_text(
        render_run_prediction_sh(
            run_dir=run_dir,
            superfold_root=superfold_root,
            python_bin=args.python_bin,
            num_parts_default=args.num_parts,
            cpus_per_task=args.cpus_per_task,
            extra_args=extra_args,
        ),
        encoding="utf-8",
    )
    os.chmod(run_prediction_sh, 0o755)

    run_prediction_local_sh = scripts_dir / "run_prediction_local.sh"
    run_prediction_local_sh.write_text(render_local_wrapper_sh(run_dir, args.num_parts), encoding="utf-8")
    os.chmod(run_prediction_local_sh, 0o755)

    readme = run_dir / "README_local.md"
    readme.write_text(
        "\n".join(
            [
                "# SuperFold run directory",
                "",
                "This folder was generated by `superfold/001_prepare_superfold_run.py`.",
                "",
                "## Structure",
                "- `inputs/`: prepared inputs + `file_list.txt`",
                "- `outputs/`: SuperFold outputs",
                "- `scripts/`: run scripts",
                "- `logs/`: Slurm stdout/stderr by default",
                "",
                "## Run on Slurm",
                "```bash",
                f"cd {scripts_dir}",
                "# Override array size at submit time if desired:",
                "sbatch --array=1-4 run_prediction.sh",
                "```",
                "",
                "## Run locally (sequential)",
                "```bash",
                f"cd {scripts_dir}",
                "bash run_prediction_local.sh 4",
                "```",
                "",
                "## Local smoke test (no SuperFold deps)",
                "```bash",
                f"cd {scripts_dir}",
                "SUPERFOLD_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 SLURM_ARRAY_TASK_COUNT=4 bash run_prediction.sh",
                "```",
                "",
                "## Notes",
                "- FASTA headers are sanitized to safe names; see `inputs/manifest.json` for mapping.",
                "- SuperFold uses `<superfold_root>/alphafold_weights.pth`; this script will write it when `--weights-dir` is provided.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("✓ Prepared SuperFold run directory")
    print(f"  run_dir: {run_dir}")
    print(f"  inputs:  {inputs_dir}")
    print(f"  outputs: {outputs_dir}")
    print(f"  scripts: {scripts_dir}")
    print(f"  files:   {len(rel_paths)} (see inputs/file_list.txt)")
    if weights_pointer:
        print(f"  weights: {weights_pointer}")
    else:
        print("  weights: (not set) - create superfold/alphafold_weights.pth or pass --weights-dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
