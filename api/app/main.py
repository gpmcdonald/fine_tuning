from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.app.routes.health import router as health_router
from api.app.routes.chat import router as chat_router
from api.app.routes.image import router as image_router

app = FastAPI(title="SyMoNeuRaL Fine Tuning API")

# CORS (fine for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- API routes FIRST (critical) ----
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(image_router)

# ---- Paths ----
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../fine_tuning
FRONTEND_DIR = REPO_ROOT / "api" / "frontend"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# ---- Static mounts AFTER routers (critical) ----
# Outputs at /outputs
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Frontend at /
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")