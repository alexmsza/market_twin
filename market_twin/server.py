"""
Market Twin • Servidor FastAPI & Dashboard Executivo de Validação
"""

from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from market_twin.models import (
    PricingModel,
    ProjectCategory,
    ProjectPitch,
    Region,
    SimulationResult,
)
from market_twin.report_generator import ExecutiveReportGenerator
from market_twin.simulator import MarketTwinSimulator

app = FastAPI(
    title="Market Twin API",
    description="Simulador de População Sintética para Validação de PMF e Risco de Investimento",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")
STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

simulator = MarketTwinSimulator()
report_gen = ExecutiveReportGenerator()


class SimulationRequest(BaseModel):
    name: str
    tagline: str
    category: ProjectCategory = ProjectCategory.SAAS_B2B
    problem_statement: str
    solution_description: str
    price_usd_monthly: float = 29.0
    population_size: int = 10000
    use_llm: bool = False
    seed: int = 42


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": [c.value for c in ProjectCategory]},
    )


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_market(payload: SimulationRequest):
    try:
        sim = MarketTwinSimulator(seed=payload.seed)
        pitch = ProjectPitch(
            name=payload.name,
            tagline=payload.tagline,
            category=payload.category,
            problem_statement=payload.problem_statement,
            solution_description=payload.solution_description,
            price_usd_monthly=payload.price_usd_monthly,
            pricing_model=PricingModel.FIXED_SUBSCRIPTION,
        )
        result = sim.run_simulation(
            pitch=pitch,
            population_size=payload.population_size,
            use_llm_focus_group=payload.use_llm,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate/pdf")
async def download_simulation_pdf(payload: SimulationRequest):
    try:
        sim = MarketTwinSimulator(seed=payload.seed)
        pitch = ProjectPitch(
            name=payload.name,
            tagline=payload.tagline,
            category=payload.category,
            problem_statement=payload.problem_statement,
            solution_description=payload.solution_description,
            price_usd_monthly=payload.price_usd_monthly,
            pricing_model=PricingModel.FIXED_SUBSCRIPTION,
        )
        result = sim.run_simulation(
            pitch=pitch,
            population_size=payload.population_size,
            use_llm_focus_group=payload.use_llm,
        )
        pdf_bytes = report_gen.generate_pdf(result)

        filename = f"dossie_market_twin_{payload.name.lower().replace(' ', '_')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "app": "market_twin", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("market_twin.server:app", host="127.0.0.1", port=8090, reload=True)
