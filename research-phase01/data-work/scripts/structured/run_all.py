"""Run the whole structuring pipeline in order.

    python research-phase01\\data-work\\scripts\\structured\\run_all.py
    python ...\\run_all.py --force       # re-call the API (costs money, ~$0.08)

Stages are individually idempotent; with the cache present a full rerun costs $0.
"""
import argparse, subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
STAGES = ["segments.py", "stage1_dishes.py", "stage2_facts.py", "stage3_db.py", "stage4_report.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in STAGES:
        cmd = [sys.executable, str(HERE / s)]
        if a.force and s in ("stage1_dishes.py", "stage2_facts.py"):
            cmd.append("--force")
        print(f"=== {s}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(HERE.parents[3]))


if __name__ == "__main__":
    main()
