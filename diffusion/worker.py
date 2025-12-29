# diffusion/worker.py
from __future__ import annotations

import json
import os
import socket
import time
import traceback
from pathlib import Path

from diffusion import pipeline

# Import DB from API package via PYTHONPATH=.
from api.app.db import database as db


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    return f"{host}:{pid}"


def _outputs_url_for_path(out_path: str) -> str:
    # API mounts /outputs/images -> outputs/images
    name = Path(out_path).name
    return f"/outputs/images/{name}"


def main() -> None:
    verbose = _env_flag("SYM_WORKER_VERBOSE", "1")
    warmup = _env_flag("SYM_PIPE_WARMUP", "1")
    poll_s = float(os.environ.get("SYM_WORKER_POLL_S", "1.0"))

    wid = _worker_id()

    if verbose:
        info = pipeline.describe()
        print(
            "[worker] starting\n"
            f"  worker_id: {wid}\n"
            f"  model_id:  {os.environ.get('SYM_SD_MODEL', pipeline.DEFAULT_MODEL)}\n"
            f"  cuda:      {info['cuda_available']}\n"
            f"  gpu:       {info['gpu']}\n"
            f"  out_dir:   {info['out_dir']}\n"
            f"  warmup:    {warmup}\n"
            f"  poll_s:    {poll_s}\n",
            flush=True,
        )

    # Optional warmup: load model immediately so first job is fast
    if warmup:
        try:
            os.environ["SYM_WORKER_VERBOSE"] = "1" if verbose else "0"
            w = pipeline.warmup()
            if verbose:
                print(f"[worker] warmup complete: loaded={w.get('loaded')} device={w.get('device')}", flush=True)
        except Exception:
            print("[worker] warmup failed:", flush=True)
            traceback.print_exc()

    if verbose:
        print("[worker] ready. Polling queue… Ctrl+C to stop.", flush=True)

    while True:
        try:
            job = db.claim_next_job(wid)
            if not job:
                time.sleep(poll_s)
                continue

            job_id = job["id"]
            prompt = job["prompt"]
            params = json.loads(job["params_json"] or "{}")

            if verbose:
                pshort = (prompt[:120] + "…") if len(prompt) > 120 else prompt
                print(f"[worker] claimed {job_id} :: {pshort}", flush=True)

            # Keep lease alive during long runs
            last_heartbeat = time.time()

            def hb():
                nonlocal last_heartbeat
                now = time.time()
                if now - last_heartbeat > 10:
                    db.heartbeat_job(job_id, wid)
                    last_heartbeat = now

            # Pull params with safe defaults
            model_id = params.get("model_id", os.environ.get("SYM_SD_MODEL", pipeline.DEFAULT_MODEL))
            steps = int(params.get("steps", 28))
            guidance_scale = float(params.get("guidance_scale", 7.5))
            width = int(params.get("width", 512))
            height = int(params.get("height", 512))
            seed = params.get("seed", None)
            device = os.environ.get("SYM_DEVICE", None)  # "cuda"|"cpu"|None

            hb()
            out_path = pipeline.generate_image(
                prompt,
                model_id=model_id,
                device=device,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
            )
            hb()

            out_url = _outputs_url_for_path(out_path)
            db.complete_job(job_id, wid, out_path, out_url)

            if verbose:
                print(f"[worker] done {job_id} -> {out_url}", flush=True)

        except KeyboardInterrupt:
            print("\n[worker] stopping.", flush=True)
            return
        except Exception as e:
            # If we have a current job context, fail it; otherwise just log
            try:
                if "job_id" in locals():
                    db.fail_job(locals()["job_id"], wid, str(e))
            except Exception:
                pass

            print("[worker] ERROR:", flush=True)
            traceback.print_exc()
            time.sleep(1.0)


if __name__ == "__main__":
    main()