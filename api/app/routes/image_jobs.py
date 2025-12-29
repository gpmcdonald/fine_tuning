# api/app/routes/image_jobs.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.app.db.database import create_image_job, ensure_user, get_job, list_jobs_for_user

router = APIRouter(prefix="/image", tags=["image-jobs"])


def _anon_id(request: Request) -> Optional[str]:
    return request.cookies.get("sym_anon")


class CreateImageJobIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=28, ge=1, le=80)
    guidance: float = Field(default=7.5, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=256, le=1536)
    height: int = Field(default=512, ge=256, le=1536)
    seed: Optional[int] = None
    model_id: Optional[str] = None  # if None worker uses env/default


@router.post("/jobs")
def create_job(req: Request, data: CreateImageJobIn) -> Dict[str, Any]:
    anon = _anon_id(req)
    if not anon:
        raise HTTPException(status_code=400, detail="missing anon id cookie (sym_anon)")

    ensure_user(anon)

    params = {
        "steps": data.steps,
        "guidance_scale": data.guidance,
        "width": data.width,
        "height": data.height,
        "seed": data.seed,
    }
    if data.model_id:
        params["model_id"] = data.model_id

    try:
        job = create_image_job(anon, data.prompt, params)
    except RuntimeError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True, "job_id": job["id"], "status": job["status"]}


@router.get("/jobs/{job_id}")
def job_status(req: Request, job_id: str) -> Dict[str, Any]:
    anon = _anon_id(req)
    if not anon:
        raise HTTPException(status_code=400, detail="missing anon id cookie (sym_anon)")

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job["user_id"] != anon:
        raise HTTPException(status_code=403, detail="not your job")

    return {
        "ok": True,
        "job": {
            "id": job["id"],
            "status": job["status"],
            "prompt": job["prompt"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "output_url": job["output_url"],
            "error": job["error"],
        },
    }


@router.get("/jobs")
def my_jobs(req: Request, limit: int = 25) -> Dict[str, Any]:
    anon = _anon_id(req)
    if not anon:
        raise HTTPException(status_code=400, detail="missing anon id cookie (sym_anon)")

    jobs = list_jobs_for_user(anon, limit=min(int(limit), 100))
    slim = []
    for j in jobs:
        slim.append(
            {
                "id": j["id"],
                "status": j["status"],
                "prompt": j["prompt"],
                "created_at": j["created_at"],
                "output_url": j["output_url"],
                "error": j["error"],
            }
        )
    return {"ok": True, "jobs": slim}