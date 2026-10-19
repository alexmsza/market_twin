"""
Market Twin • Suíte de Testes Automatizados (pytest)
Validação de integridade estatística, desempenho de inferência (< 2s para 10k personas) e geração de relatórios.
"""

from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient

from market_twin.models import PricingModel, ProjectCategory, ProjectPitch, Region
from market_twin.population_generator import SyntheticPopulationGenerator
from market_twin.report_generator import ExecutiveReportGenerator
from market_twin.server import app
from market_twin.simulator import MarketTwinSimulator


@pytest.fixture
def sample_pitch() -> ProjectPitch:
    return ProjectPitch(
        name="API de Monitoramento de Pagamentos PIX",
        tagline="Monitoramento em tempo real com failover de pagamentos",
        category=ProjectCategory.FINTECH,
        problem_statement="APIs bancárias falham sem alertas imediatos",
        solution_description="Heartbeat a cada 5 segundos com rota de contingência",
        price_usd_monthly=49.0,
        pricing_model=PricingModel.FIXED_SUBSCRIPTION,
    )


def test_population_generator_10k_distribution():
    generator = SyntheticPopulationGenerator(seed=42)
    start = time.perf_counter()
    personas = generator.generate_population(size=10000)
    duration = time.perf_counter() - start

    assert len(personas) == 10000
    assert duration < 2.0, f"Geração demorou {duration:.2f}s, esperado < 2.0s"

    # Validar sanidade de campos
    for p in personas[:100]:
        assert 18 <= p.age <= 72
        assert 0.0 <= p.tech_savviness <= 1.0
        assert p.monthly_disposable_income_usd > 0
        assert 0.0 <= p.price_sensitivity <= 1.0
        assert p.region in list(Region)

    # Validar proporção regional aproximada (LATAM ~ 35%, NA ~ 25%)
    latam_count = sum(1 for p in personas if p.region == Region.LATAM)
    na_count = sum(1 for p in personas if p.region == Region.NORTH_AMERICA)

    assert 3000 <= latam_count <= 4000, f"LATAM count {latam_count} fora do intervalo esperado"
    assert 2000 <= na_count <= 3000, f"NA count {na_count} fora do intervalo esperado"


def test_simulation_engine_performance_10k(sample_pitch: ProjectPitch):
    simulator = MarketTwinSimulator(seed=42)
    start = time.perf_counter()
    result = simulator.run_simulation(pitch=sample_pitch, population_size=10000, use_llm_focus_group=False)
    duration = time.perf_counter() - start

    assert duration < 2.5, f"Simulação de 10k demorou {duration:.2f}s, esperado < 2.5s"
    assert result.population_size == 10000
    assert 0.0 <= result.pmf_index <= 100.0
    assert -100.0 <= result.nps_score <= 100.0
    assert 0.0 <= result.conversion_rate <= 100.0
    assert result.verdict in [
        "FORTE RECOMENDAÇÃO (GO)",
        "GO CONDICIONAL (REFINAR MODELO)",
        "ALTO RISCO (CAUTION)",
        "NÃO INVESTIR (NO-GO)",
    ]
    assert len(result.breakdown_by_region) == 5
    assert len(result.focus_group) == 5


def test_price_elasticity_law_of_demand(sample_pitch: ProjectPitch):
    simulator = MarketTwinSimulator(seed=42)
    result = simulator.run_simulation(pitch=sample_pitch, population_size=2000, use_llm_focus_group=False)

    elasticity = result.price_elasticity
    assert len(elasticity) >= 4

    # A taxa de conversão deve ser decrescente ou estável conforme o preço sobe
    for i in range(len(elasticity) - 1):
        assert elasticity[i].conversion_rate >= elasticity[i + 1].conversion_rate - 0.01


def test_executive_pdf_generation(sample_pitch: ProjectPitch):
    simulator = MarketTwinSimulator(seed=42)
    result = simulator.run_simulation(pitch=sample_pitch, population_size=1000, use_llm_focus_group=False)

    generator = ExecutiveReportGenerator()
    pdf_bytes = generator.generate_pdf(result)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")


def test_api_server_endpoints(sample_pitch: ProjectPitch):
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    # API Simulate
    payload = {
        "name": sample_pitch.name,
        "tagline": sample_pitch.tagline,
        "category": sample_pitch.category.value,
        "problem_statement": sample_pitch.problem_statement,
        "solution_description": sample_pitch.solution_description,
        "price_usd_monthly": sample_pitch.price_usd_monthly,
        "population_size": 1000,
        "use_llm": False,
        "seed": 42,
    }
    res_sim = client.post("/api/simulate", json=payload)
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert sim_data["population_size"] == 1000
    assert "pmf_index" in sim_data
    assert "verdict" in sim_data

    # API PDF
    res_pdf = client.post("/api/simulate/pdf", json=payload)
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert len(res_pdf.content) > 2000
