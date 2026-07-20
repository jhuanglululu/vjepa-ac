import argparse
import json
import os
import subprocess
import sys
import time

from vjepa_ac import data
from vjepa_ac.device import pick_free_gpus


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", nargs="+", default=list(data.CAMERAS), choices=list(data.CAMERAS))
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--strides", type=int, nargs="+", default=None)
    p.add_argument("--max-gpus", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    gpus = pick_free_gpus()[: args.max_gpus]
    assert gpus, "no free GPUs found"
    print(f"free GPUs: {gpus} | cameras: {args.cameras} | seeds {args.seeds}")

    jobs = []
    for cam in args.cameras:
        cache_dir = os.path.join(data.CACHE_DIR, cam)
        assert os.path.exists(os.path.join(cache_dir, "cache.json")), (
            f"no cache for camera {cam} at {cache_dir} -- run scripts/prepare_cache.py first"
        )
        cmd = [
            sys.executable,
            "scripts/stride_gate.py",
            "--cache-dir",
            cache_dir,
            "--seeds",
            str(args.seeds),
        ]
        if args.strides:
            cmd += ["--strides"] + [str(s) for s in args.strides]
        jobs.append((cam, cmd))

    os.makedirs("records/diagnostics", exist_ok=True)
    running, queue, idle = {}, list(jobs), list(gpus)
    while queue or running:
        while queue and idle:
            cam, cmd = queue.pop(0)
            gpu = idle.pop(0)
            log = f"records/diagnostics/stride_gate_{cam}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[proc.pid] = (cam, gpu, proc, log)
            print(f"launched {cam} on gpu {gpu} (pid {proc.pid}) -> {log}")
        time.sleep(5)
        for pid in list(running):
            cam, gpu, proc, log = running[pid]
            if proc.poll() is None:
                continue
            del running[pid]
            idle.append(gpu)
            status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"{cam}: {status}")
            if proc.returncode != 0:
                with open(log) as f:
                    print("".join(f.readlines()[-15:]))

    print("\n=== combined verdicts ===")
    for cam in args.cameras:
        path = f"records/diagnostics/stride_gate_{cam}.json"
        if not os.path.exists(path):
            print(f"{cam}: no result file")
            continue
        with open(path) as f:
            r = json.load(f)
        passing = r.get("passing_strides", [])
        print(f"{cam}: passing strides {passing or 'NONE'}")
        for row in r["rows"]:
            print(
                f"  stride {row['stride']:>2} | pair {row['pair_test_motion']:+.3f} "
                f"±{row['pair_se']:.3f} | margin {row['margin']:+.3f} ±{row['margin_se']:.3f} | "
                f"train {row['pair_train_motion']:+.3f} | "
                + ("PASS" if row["passed"] else "fail")
                + (" (probe underfit)" if row["probe_limited"] else "")
            )


if __name__ == "__main__":
    main()
