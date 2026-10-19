"""
Market Twin • Benchmark de Altíssima Performance (Motor Nativo Rust Rayon)
Verifica tempo de inferência para 10.000, 50.000 e 100.000 personas sintéticas.
"""

from __future__ import annotations

import time
import pytest

from market_twin.models import PricingModel, ProjectCategory, ProjectPitch
from market_twin.native_engine import is_native_available
from market_twin.simulator import MarketTwinSimulator


def test_native_engine_is_loaded():
    assert is_native_available(), "O motor nativo Rust Rayon (market_twin_engine.dll) deve estar compilado e carregado!"


@pytest.mark.parametrize("pop_size", [10000, 50000, 100000])
def test_simulation_speed_scales_subsecond(pop_size: int):
    simulator = MarketTwinSimulator(seed=42)
    pitch = ProjectPitch(
        name="Plataforma de Pagamentos Globais",
        tagline="Orquestração de pagamentos com roteamento de menor taxa",
        category=ProjectCategory.FINTECH,
        problem_statement="Perda de conversão em pagamentos internacionais por recusa bancária",
        solution_description="Roteamento inteligente de transações multi-adquirente com fallback instantâneo",
        price_usd_monthly=79.0,
        pricing_model=PricingModel.FIXED_SUBSCRIPTION,
    )

    start = time.perf_counter()
    result = simulator.run_simulation(pitch=pitch, population_size=pop_size, use_llm_focus_group=False)
    duration = time.perf_counter() - start

    print(f"\n[BENCHMARK] {pop_size:,} personas simuladas em: {duration*1000:.2f}ms (Engine Duration: {result.simulation_duration_seconds*1000:.1f}ms)")

    assert result.population_size == pop_size
    assert duration < 0.50, f"Simulação de {pop_size:,} demorou {duration:.3f}s, esperado < 0.50s!"
    assert result.conversion_rate > 0.0
    assert result.pmf_index > 0.0
