"""
Find the most-advanced training checkpoint under a given prefix (local or GCS)
and print its path, or exit with a non-zero status if none exists.

Used by the TPU startup script to auto-resume after a spot preemption: the
checkpoint filename embeds the run's start timestamp
(AbaloneTrainerSync._save_checkpoint: "{base_path}_{timestamp}_{prefix}.pkl"),
which changes on every restart, so "most recently modified file" is not a
reliable way to find "most trained" -- the iteration number encoded in the
filename (iterN / ref_iterN) is what actually orders checkpoints by progress.

Usage:
    python scripts/find_latest_checkpoint.py gs://bucket/checkpoints/model
    python scripts/find_latest_checkpoint.py checkpoints/model_cpu
"""
import argparse
import glob
import re
import subprocess
import sys

# Matches "..._iter42.pkl" and "..._ref_iter42.pkl" (not "..._final.pkl", which
# carries no iteration number -- see the module docstring for why that's fine
# here: a "final" checkpoint means that run already completed).
CHECKPOINT_RE = re.compile(r'_(?:ref_iter|iter)(\d+)\.pkl$')


def list_checkpoint_files(checkpoint_path_prefix):
    pattern = f"{checkpoint_path_prefix}_*_*.pkl"
    if checkpoint_path_prefix.startswith("gs://"):
        result = subprocess.run(
            ["gsutil", "ls", pattern],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return glob.glob(pattern)


def find_latest_checkpoint(checkpoint_path_prefix):
    """Return the path of the checkpoint with the highest iteration number, or None."""
    best_path, best_iter = None, -1
    for path in list_checkpoint_files(checkpoint_path_prefix):
        match = CHECKPOINT_RE.search(path)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration > best_iter:
            best_iter = iteration
            best_path = path
    return best_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'checkpoint_path_prefix',
        help="Same value as --checkpoint-path passed to main.py, "
             "e.g. gs://bucket/checkpoints/model",
    )
    args = parser.parse_args()

    latest = find_latest_checkpoint(args.checkpoint_path_prefix)
    if latest is None:
        sys.exit(1)

    print(latest)


if __name__ == "__main__":
    main()
