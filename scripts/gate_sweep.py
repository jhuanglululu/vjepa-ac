import argparse
import json
import os
import subprocess
import sys
import time

import stride_gate

from vjepa_ac import data
from vjepa_ac.device import pick_free_gpus

MAX_GPUS = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cameras", nargs="+", default=list(data.CAMERAS), choices=list(data.CAMERAS))
    p.add_argument("--strides", type=int, nargs="+", default=list(stride_gate.SWEEP_STRIDES))
    p.add_argument("--seeds", type=int, default=stride_gate.SWEEP_SEEDS)
    return p.parse_args()


def tail_status(log):
    try:
        size = os.path.getsize(log)
        with open(log, "rb") as f:
            f.seek(max(0, size - 4096))
            text = f.read().decode(errors="replace")
    except OSError:
        return "no output yet"
    parts = [seg.strip() for line in text.splitlines() for seg in line.split("\r") if seg.strip()]
    return parts[-1][:110] if parts else "no output yet"


def main():
    args = parse_args()
    cameras = args.cameras
    gpus = pick_free_gpus()[:MAX_GPUS]
    assert gpus, "no free GPUs found"
    print(f"free GPUs: {gpus} | cameras: {cameras} | strides {args.strides} | seeds {args.seeds}")

    jobs = []
    for cam in cameras:
        cache_dir = data.camera_cache_dir(cam)
        assert os.path.exists(os.path.join(cache_dir, "cache.json")), (
            f"no cache for camera {cam} at {cache_dir} -- run scripts/prepare_cache.py first"
        )
        cmd = [sys.executable, "scripts/stride_gate.py", cam, str(args.seeds)]
        cmd += [str(s) for s in args.strides]
        jobs.append((cam, cmd))

    os.makedirs("records/diagnostics", exist_ok=True)
    running, queue, idle = {}, list(jobs), list(gpus)
    last_status = time.monotonic()
    while queue or running:
        while queue and idle:
            cam, cmd = queue.pop(0)
            gpu = idle.pop(0)
            log = f"records/diagnostics/stride_gate_{cam}.log"
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            phys = cvd.split(",")[gpu] if cvd else str(gpu)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=phys, PYTHONUNBUFFERED="1")
            proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[proc.pid] = (cam, gpu, proc, log)
            print(f"launched {cam} on gpu {gpu} (pid {proc.pid}) -> {log}")
        time.sleep(5)
        if running and time.monotonic() - last_status >= 30:
            last_status = time.monotonic()
            for cam, gpu, proc, log in running.values():
                print(f"  [{cam} gpu{gpu}] {tail_status(log)}", flush=True)
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
    for cam in cameras:
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
