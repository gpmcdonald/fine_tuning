from __future__ import annotations

import os
import time
import traceback

from diffusion import pipeline


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")


def main() -> None:
    verbose = _env_flag("SYM_WORKER_VERBOSE", "1")
    warmup = _env_flag("SYM_PIPE_WARMUP", "0")

    # Banner
    info = pipeline.describe()
    if verbose:
        print(
            "[worker] starting\n"
            f"  model_id: {os.environ.get('SYM_SD_MODEL', pipeline.DEFAULT_MODEL)}\n"
            f"  cuda:     {info['cuda_available']}\n"
            f"  gpu:      {info['gpu']}\n"
            f"  out_dir:  {info['out_dir']}\n"
            f"  warmup:   {warmup}\n",
            flush=True,
        )

    # Optional warmup (load model now, not on first request)
    if warmup:
        try:
            os.environ["SYM_WORKER_VERBOSE"] = "1" if verbose else os.environ.get("SYM_WORKER_VERBOSE", "0")
            w = pipeline.warmup()
            if verbose:
                print(f"[worker] warmup complete: loaded={w.get('loaded')} device={w.get('device')}", flush=True)
        except Exception:
            print("[worker] warmup failed:", flush=True)
            traceback.print_exc()

    # Stay alive (simple daemon loop)
    if verbose:
        print("[worker] ready (idle loop). Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        if verbose:
            print("\n[worker] stopping", flush=True)


if __name__ == "__main__":
    main()
