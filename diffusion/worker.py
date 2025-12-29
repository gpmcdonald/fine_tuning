# diffusion/worker.py
from __future__ import annotations

import os
import time
import traceback

from diffusion import pipeline
from diffusion import queue


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")


def main() -> None:
    verbose = _env_flag("SYM_WORKER_VERBOSE", "1")
    warmup = _env_flag("SYM_PIPE_WARMUP", "0")
    poll_s = float(os.environ.get("SYM_WORKER_POLL_S", "1.0"))

    # Banner
    info = pipeline.describe()
    if verbose:
        print(
            "[worker] starting\n"
            f"  model_id: {os.environ.get('SYM_SD_MODEL', pipeline.DEFAULT_MODEL)}\n"
            f"  cuda:     {info['cuda_available']}\n"
            f"  gpu:      {info['gpu']}\n"
            f"  out_dir:  {info['out_dir']}\n"
            f"  queue:    {os.environ.get('SYM_QUEUE_DIR', 'queue')}\n"
            f"  warmup:   {warmup}\n",
            flush=True,
        )

    # Optional warmup
    if warmup:
        try:
            os.environ["SYM_WORKER_VERBOSE"] = "1" if verbose else os.environ.get("SYM_WORKER_VERBOSE", "0")
            w = pipeline.warmup()
            if verbose:
                print(f"[worker] warmup complete: loaded={w.get('loaded')} device={w.get('device')}", flush=True)
        except Exception:
            print("[worker] warmup failed:", flush=True)
            traceback.print_exc()

    if verbose:
        print("[worker] ready. Watching queue/pending ... Ctrl+C to stop.", flush=True)

    try:
        while True:
            job_path = queue.claim_next()
            if not job_path:
                time.sleep(poll_s)
                continue

            job = queue.read_json(job_path)
            job_id = job.get("id", job_path.stem)
            prompt = (job.get("prompt") or "").strip()
            params = job.get("params") or {}

            if verbose:
                print(f"[worker] claimed job {job_id}: {prompt[:80]}", flush=True)

            try:
                # Pull optional params with safe defaults
                model_id = params.get("model_id", os.environ.get("SYM_SD_MODEL", pipeline.DEFAULT_MODEL))
                steps = int(params.get("steps", 28))
                guidance = float(params.get("guidance_scale", 7.5))
                width = int(params.get("width", 512))
                height = int(params.get("height", 512))
                seed = params.get("seed", None)

                out_path = pipeline.generate_image(
                    prompt,
                    model_id=model_id,
                    steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    seed=seed,
                )

                queue.complete(job_path, {"id": job_id, "prompt": prompt, "image_path": out_path})
                if verbose:
                    print(f"[worker] done job {job_id}: {out_path}", flush=True)

            except Exception as e:
                queue.fail(job_path, f"{type(e).__name__}: {e}")
                print(f"[worker] failed job {job_id}", flush=True)
                traceback.print_exc()

    except KeyboardInterrupt:
        if verbose:
            print("\n[worker] stopping", flush=True)


if __name__ == "__main__":
    main()