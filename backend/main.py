import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.models import init_db
from backend.routers import alerts, analyze, market, schemes, soil, voice, whatsapp

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

app = FastAPI(
    title="Kisaan AI",
    description=(
        "Autonomous multimodal farm intelligence — crop disease diagnosis, "
        "market prices, soil health, govt schemes, and multilingual voice advisory."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in [analyze.router, market.router, voice.router,
               soil.router, schemes.router, alerts.router, whatsapp.router]:
    app.include_router(router)


@app.on_event("startup")
async def startup():
    await init_db()
    logging.getLogger(__name__).info("Kisaan AI v2.0 ready.")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "kisaan-ai", "version": "2.0.0"}
