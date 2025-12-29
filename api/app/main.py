# api/app/main.py
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from api.app.db.database import init_db
from api.app.routes.health import router as health_router
from api.app.routes.chat import router as chat_router
from api.app.routes.image_jobs import router as image_jobs_router

APP_NAME = os.environ.get("SYM_APP_NAME", "SyMoNeuRaL Image Generator")
REPO_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = Path(os.environ.get("SYM_FRONTEND_DIR", str(REPO_ROOT / "frontend")))
OUTPUTS_DIR = Path(os.environ.get("SYM_OUT_DIR", str(REPO_ROOT.parent / "outputs" / "images")))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ANON_COOKIE = "sym_anon"


class AnonCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.cookies.get(ANON_COOKIE):
            return response

        anon_id = str(uuid.uuid4())

        # Secure cookie only when HTTPS (behind Cloudflare this usually is https)
        scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme or "").lower()
        secure_cookie = scheme == "https"

        response.set_cookie(
            ANON_COOKIE,
            anon_id,
            httponly=True,
            secure=secure_cookie,
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,
            path="/",
        )
        return response


app = FastAPI(title=APP_NAME, version="0.2.0")

app.add_middleware(AnonCookieMiddleware)

# Init DB on boot
init_db()

# Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(image_jobs_router)

# Static mounts
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="static")

# Serve generated images
app.mount("/outputs/images", StaticFiles(directory=str(OUTPUTS_DIR), html=False), name="outputs-images")


@app.get("/", include_in_schema=False)
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"ok": False, "error": "frontend missing", "expected": str(index_path)}, status_code=500)
    return FileResponse(str(index_path))