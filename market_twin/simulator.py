"""
Market Twin • Motor de Simulação em Dois Níveis (Dual-Tier Cognitive Engine)
Tier 1: Modelagem Micro-Econômica Vetorizada de Monte Carlo (10.000+ Personas)
Tier 2: Focus Group Cognitivo com SLM/LLM Local (Ollama / aszadev-pro)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests

from market_twin.models import (
    CohortFocusGroupFeedback,
    Persona,
    PersonaArchetype,
    PersonaEvaluation,
    PricePointDemand,
    ProjectCategory,
    ProjectPitch,
    Region,
    RegionalMetric,
    SimulationResult,
)
from market_twin.population_generator import SyntheticPopulationGenerator

logger = logging.getLogger("market_twin.simulator")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_SLM_MODEL = "aszadev-pro:latest"
FALLBACK_FAST_MODEL = "aszadev-fast:latest"


# Matriz de Afinidade Base (Categoria de Projeto x Arquétipo de Persona)
CATEGORY_AFFINITY: Dict[ProjectCategory, Dict[PersonaArchetype, float]] = {
    ProjectCategory.SAAS_B2B: {
        PersonaArchetype.ENTERPRISE_BUYER: 0.90,
        PersonaArchetype.SMB_FOUNDER: 0.85,
        PersonaArchetype.FREELANCER_DEV: 0.65,
        PersonaArchetype.TECH_ENTHUSIAST: 0.50,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.15,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.20,
    },
    ProjectCategory.DEVTOOL: {
        PersonaArchetype.FREELANCER_DEV: 0.95,
        PersonaArchetype.TECH_ENTHUSIAST: 0.85,
        PersonaArchetype.SMB_FOUNDER: 0.60,
        PersonaArchetype.ENTERPRISE_BUYER: 0.65,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.55,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.10,
    },
    ProjectCategory.AI_AGENT: {
        PersonaArchetype.SMB_FOUNDER: 0.92,
        PersonaArchetype.TECH_ENTHUSIAST: 0.90,
        PersonaArchetype.FREELANCER_DEV: 0.85,
        PersonaArchetype.ENTERPRISE_BUYER: 0.75,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.60,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.65,
    },
    ProjectCategory.FINTECH: {
        PersonaArchetype.SMB_FOUNDER: 0.90,
        PersonaArchetype.ENTERPRISE_BUYER: 0.85,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.75,
        PersonaArchetype.FREELANCER_DEV: 0.70,
        PersonaArchetype.TECH_ENTHUSIAST: 0.65,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.50,
    },
    ProjectCategory.CONSUMER_APP: {
        PersonaArchetype.EVERYDAY_CONSUMER: 0.90,
        PersonaArchetype.TECH_ENTHUSIAST: 0.80,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.85,
        PersonaArchetype.FREELANCER_DEV: 0.60,
        PersonaArchetype.SMB_FOUNDER: 0.40,
        PersonaArchetype.ENTERPRISE_BUYER: 0.25,
    },
    ProjectCategory.EDTECH: {
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.95,
        PersonaArchetype.FREELANCER_DEV: 0.80,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.70,
        PersonaArchetype.TECH_ENTHUSIAST: 0.65,
        PersonaArchetype.SMB_FOUNDER: 0.50,
        PersonaArchetype.ENTERPRISE_BUYER: 0.45,
    },
    ProjectCategory.ECOMMERCE: {
        PersonaArchetype.EVERYDAY_CONSUMER: 0.85,
        PersonaArchetype.SMB_FOUNDER: 0.80,
        PersonaArchetype.TECH_ENTHUSIAST: 0.60,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.70,
        PersonaArchetype.FREELANCER_DEV: 0.50,
        PersonaArchetype.ENTERPRISE_BUYER: 0.40,
    },
    ProjectCategory.HARDWARE_IOT: {
        PersonaArchetype.TECH_ENTHUSIAST: 0.92,
        PersonaArchetype.ENTERPRISE_BUYER: 0.70,
        PersonaArchetype.SMB_FOUNDER: 0.65,
        PersonaArchetype.FREELANCER_DEV: 0.60,
        PersonaArchetype.EVERYDAY_CONSUMER: 0.50,
        PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.35,
    },
}


class MarketTwinSimulator:
    """
    Simulador híbrido: Estatística Vetorizada Monte Carlo + LLM/SLM Focus Group.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pop_generator = SyntheticPopulationGenerator(seed=seed)

    def run_simulation(
        self,
        pitch: ProjectPitch,
        population_size: int = 10000,
        use_llm_focus_group: bool = True,
    ) -> SimulationResult:
        """
        Executa a simulação completa para uma população de N clones virtuais.
        """
        start_time = time.perf_counter()

        # 1. Gerar População Sintética
        personas = self.pop_generator.generate_population(size=population_size)

        # 2. Avaliação Micro-Econômica Vetorizada (Tier 1)
        evaluations = self._evaluate_population(personas, pitch)

        # 3. Consolidação Estatística de Métricas
        pmf_index, nps_score, conversion_rate = self._compute_aggregate_metrics(evaluations)

        # 4. Análise por Macrorregião
        breakdown_by_region = self._compute_regional_breakdown(personas, evaluations)

        # 5. Curva de Elasticidade de Preço (Van Westendorp PSM)
        price_elasticity = self._compute_price_elasticity(personas, pitch)

        # 6. Veredito Executivo de Investimento
        verdict, badge, rationale = self._determine_investment_verdict(
            pmf_index=pmf_index,
            nps=nps_score,
            conversion=conversion_rate,
            pitch=pitch,
        )

        # 7. Focus Group Qualitativo Estratificado (Tier 2 via SLM/LLM)
        focus_group = self._run_cohort_focus_group(
            personas=personas,
            evaluations=evaluations,
            pitch=pitch,
            use_llm=use_llm_focus_group,
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        return SimulationResult(
            project_name=pitch.name,
            population_size=population_size,
            pmf_index=round(pmf_index, 1),
            nps_score=round(nps_score, 1),
            conversion_rate=round(conversion_rate, 2),
            verdict=verdict,
            verdict_badge=badge,
            verdict_rationale=rationale,
            breakdown_by_region=breakdown_by_region,
            price_elasticity=price_elasticity,
            focus_group=focus_group,
            simulation_duration_seconds=elapsed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _evaluate_population(
        self,
        personas: List[Persona],
        pitch: ProjectPitch,
    ) -> List[PersonaEvaluation]:
        """
        Executa a função de utilidade micro-econômica para cada indivíduo.
        """
        category_weights = CATEGORY_AFFINITY.get(
            pitch.category,
            CATEGORY_AFFINITY[ProjectCategory.SAAS_B2B],
        )

        evaluations: List[PersonaEvaluation] = []
        rng = np.random.default_rng(self.seed)

        target_region_set = set(pitch.target_regions)

        for p in personas:
            # Penalidade se a região não for alvo
            region_multiplier = 1.0 if p.region in target_region_set else 0.25

            # Afinidade arquetípica com o produto
            affinity = category_weights.get(p.archetype, 0.5)

            # Cálculo de Disposição Máxima a Pagar (WTP em USD)
            # WTP escala com a renda disponível, reduzida pela sensibilidade a preço
            base_wtp = (p.monthly_disposable_income_usd * 0.05) * affinity
            income_factor = np.log1p(p.monthly_disposable_income_usd / 500.0)
            max_wtp = float(
                np.clip(
                    base_wtp * income_factor * (1.2 - p.price_sensitivity * 0.6),
                    2.0,
                    15000.0,
                )
            )

            # Relação de valor percebido (Perceived Value Ratio = WTP / Preço)
            price = max(pitch.price_usd_monthly, 1.0)
            pvr = max_wtp / price

            # Função de Utilidade Estocástica U = f(Affinity, PVR, Savviness) - Friction(Skepticism)
            stochastic_noise = float(rng.normal(0, 0.15))
            utility = (
                affinity * 0.40
                + min(pvr / 2.0, 1.0) * 0.35
                + p.tech_savviness * 0.15
                - p.skepticism_level * 0.25
                + stochastic_noise
            ) * region_multiplier

            # Condição de Compra: Utilidade > Limiar (0.48) e PVR >= 0.85
            would_buy = bool(utility >= 0.48 and pvr >= 0.80)

            # Cálculo do PMF Score Individual (0 a 10)
            raw_pmf = (utility * 10.0) + (pvr * 1.5)
            pmf_score = float(np.clip(raw_pmf, 0.0, 10.0))

            # Classificação NPS
            if pmf_score >= 8.5 and would_buy:
                nps_cat = "promoter"
            elif pmf_score >= 6.5:
                nps_cat = "passive"
            else:
                nps_cat = "detractor"

            evaluations.append(
                PersonaEvaluation.model_construct(
                    persona_id=p.id,
                    would_buy=would_buy,
                    pmf_score=round(pmf_score, 2),
                    nps_category=nps_cat,
                    max_wtp_usd=round(max_wtp, 2),
                    perceived_value_ratio=round(pvr, 2),
                )
            )

        return evaluations

    def _compute_aggregate_metrics(
        self,
        evaluations: List[PersonaEvaluation],
    ) -> Tuple[float, float, float]:
        """
        Calcula PMF Index, NPS e Taxa de Conversão.
        """
        n = len(evaluations)
        if n == 0:
            return 0.0, 0.0, 0.0

        buyers = sum(1 for e in evaluations if e.would_buy)
        conversion_rate = (buyers / n) * 100.0

        promoters = sum(1 for e in evaluations if e.nps_category == "promoter")
        detractors = sum(1 for e in evaluations if e.nps_category == "detractor")

        # NPS = % Promotores - % Detratores (-100 a +100)
        nps = ((promoters - detractors) / n) * 100.0

        # PMF Index ponderado (0 a 100): Combina proporção de promotores e satisfação média
        mean_pmf_score = float(np.mean([e.pmf_score for e in evaluations]))
        pmf_index = min(100.0, (mean_pmf_score * 7.0) + (conversion_rate * 0.30))

        return pmf_index, nps, conversion_rate

    def _compute_regional_breakdown(
        self,
        personas: List[Persona],
        evaluations: List[PersonaEvaluation],
    ) -> List[RegionalMetric]:
        """
        Calcula as métricas segmentadas por macrorregião geográfica.
        """
        metrics: Dict[str, Dict[str, Any]] = {}

        for p, ev in zip(personas, evaluations):
            reg = p.region.value
            if reg not in metrics:
                metrics[reg] = {
                    "total": 0,
                    "buyers": 0,
                    "pmf_sum": 0.0,
                    "wtp_sum": 0.0,
                }
            metrics[reg]["total"] += 1
            if ev.would_buy:
                metrics[reg]["buyers"] += 1
            metrics[reg]["pmf_sum"] += ev.pmf_score
            metrics[reg]["wtp_sum"] += ev.max_wtp_usd

        result: List[RegionalMetric] = []
        for reg_name, data in metrics.items():
            tot = data["total"]
            cr = (data["buyers"] / tot * 100.0) if tot > 0 else 0.0
            avg_pmf = (data["pmf_sum"] / tot) if tot > 0 else 0.0
            avg_wtp = (data["wtp_sum"] / tot) if tot > 0 else 0.0
            result.append(
                RegionalMetric(
                    region=reg_name,
                    total_population=tot,
                    converted_users=data["buyers"],
                    conversion_rate=round(cr, 2),
                    average_pmf=round(avg_pmf, 2),
                    average_wtp_usd=round(avg_wtp, 2),
                )
            )

        # Ordenar por taxa de conversão decrescente
        result.sort(key=lambda r: r.conversion_rate, reverse=True)
        return result

    def _compute_price_elasticity(
        self,
        personas: List[Persona],
        pitch: ProjectPitch,
    ) -> List[PricePointDemand]:
        """
        Gera curva de sensibilidade de preço (Van Westendorp Price Sensitivity).
        """
        base_price = max(pitch.price_usd_monthly, 5.0)
        multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

        curve: List[PricePointDemand] = []
        max_theoretical_rev = 1.0

        raw_points = []
        for m in multipliers:
            p_usd = round(base_price * m, 2)
            # Reavaliar conversão aproximada para este preço
            converted = 0
            for p in personas:
                # WTP simplificada
                category_weights = CATEGORY_AFFINITY.get(
                    pitch.category, CATEGORY_AFFINITY[ProjectCategory.SAAS_B2B]
                )
                affinity = category_weights.get(p.archetype, 0.5)
                base_wtp = (p.monthly_disposable_income_usd * 0.05) * affinity
                income_factor = np.log1p(p.monthly_disposable_income_usd / 500.0)
                max_wtp = base_wtp * income_factor * (1.2 - p.price_sensitivity * 0.6)

                if p_usd <= max_wtp and (max_wtp / p_usd) >= 0.85:
                    converted += 1

            cr = (converted / len(personas)) * 100.0
            projected_rev = converted * p_usd
            raw_points.append((p_usd, cr, projected_rev))
            if projected_rev > max_theoretical_rev:
                max_theoretical_rev = projected_rev

        for p_usd, cr, rev in raw_points:
            rev_idx = round((rev / max_theoretical_rev) * 100.0, 1)
            curve.append(
                PricePointDemand(
                    price_usd=p_usd,
                    conversion_rate=round(cr, 2),
                    projected_revenue_index=rev_idx,
                )
            )

        return curve

    def _determine_investment_verdict(
        self,
        pmf_index: float,
        nps: float,
        conversion: float,
        pitch: ProjectPitch,
    ) -> Tuple[str, str, str]:
        """
        Determina o veredito executivo com base em benchmarks de VC/Investimento.
        """
        if pmf_index >= 65.0 and conversion >= 18.0 and nps >= 20.0:
            verdict = "FORTE RECOMENDAÇÃO (GO)"
            badge = "success"
            rationale = (
                f"O projeto '{pitch.name}' demonstrou tração excepcional na simulação. Com "
                f"PMF Index de {pmf_index:.1f}/100 e conversão de {conversion:.1f}%, a proposta de valor "
                f"supera os benchmarks médios de mercado. A disposição a pagar é robusta e a demanda "
                f"orgânica justifica aporte imediato de investimento e expansão em escala."
            )
        elif pmf_index >= 48.0 and conversion >= 10.0 and nps >= -10.0:
            verdict = "GO CONDICIONAL (REFINAR MODELO)"
            badge = "warning"
            rationale = (
                f"O projeto '{pitch.name}' possui proposta válida e interesse demonstrado (PMF {pmf_index:.1f}/100), "
                f"porém enfrenta fricções de preço ou maturidade técnica em certas regiões. "
                f"Recomenda-se realizar pilotos em praças de maior afinidade antes de investir "
                f"pesadamente em aquisição global."
            )
        elif pmf_index >= 32.0 and conversion >= 5.0:
            verdict = "ALTO RISCO (CAUTION)"
            badge = "danger"
            rationale = (
                f"A simulação aponta resistência significativa do público-alvo (PMF {pmf_index:.1f}/100, NPS {nps:.1f}). "
                f"O custo de aquisição (CAC) projetado tenderá a exceder o LTV no preço atual (${pitch.price_usd_monthly}/mês). "
                f"Recomenda-se pivotar a solução ou focar exclusivamente em um nicho de menor ceticismo."
            )
        else:
            verdict = "NÃO INVESTIR (NO-GO)"
            badge = "danger"
            rationale = (
                f"Veredito desfavorável: Rejeição severa observada na população sintética (Conversão de {conversion:.1f}% e PMF {pmf_index:.1f}/100). "
                f"A dor alegada não possui urgência suficiente para gerar disposição de pagamento sustentável. "
                f"Investir capital nesta configuração atual representa alto risco de perda integral."
            )

        return verdict, badge, rationale

    def _run_cohort_focus_group(
        self,
        personas: List[Persona],
        evaluations: List[PersonaEvaluation],
        pitch: ProjectPitch,
        use_llm: bool = True,
    ) -> List[CohortFocusGroupFeedback]:
        """
        Executa focus group qualitativo orquestrado via SLM local ou síntese contextual.
        """
        # Selecionar 5 arquétipos representativos com diversidade geográfica
        target_cohorts = [
            (PersonaArchetype.ENTERPRISE_BUYER, Region.NORTH_AMERICA),
            (PersonaArchetype.SMB_FOUNDER, Region.LATAM),
            (PersonaArchetype.FREELANCER_DEV, Region.EUROPE),
            (PersonaArchetype.TECH_ENTHUSIAST, Region.APAC),
            (PersonaArchetype.BUDGET_CONSCIOUS_STUDENT, Region.MEA),
        ]

        feedbacks: List[CohortFocusGroupFeedback] = []

        for arch, reg in target_cohorts:
            # Encontrar persona correspondente
            candidate = next(
                (p for p in personas if p.archetype == arch and p.region == reg),
                personas[0],
            )
            ev = next((e for e in evaluations if e.persona_id == candidate.id), evaluations[0])

            feedback = None
            if use_llm:
                feedback = self._query_ollama_focus_persona(candidate, ev, pitch)

            if not feedback:
                feedback = self._generate_contextual_focus_feedback(candidate, ev, pitch)

            feedbacks.append(feedback)

        return feedbacks

    def _query_ollama_focus_persona(
        self,
        persona: Persona,
        evaluation: PersonaEvaluation,
        pitch: ProjectPitch,
    ) -> Optional[CohortFocusGroupFeedback]:
        """
        Consulta o modelo Ollama local (aszadev-pro ou aszadev-fast) para gerar a fala em primeira pessoa.
        """
        prompt = (
            f"Você é uma pessoa real chamada {persona.name}, com {persona.age} anos, vivendo em {persona.country} ({persona.region.value}). "
            f"Seu perfil profissional é {persona.archetype.value}. Sua renda disponível mensal é de ${persona.monthly_disposable_income_usd:.0f} USD. "
            f"Seu nível de ceticismo é {int(persona.skepticism_level * 100)}% e sua dor diária é: '{persona.primary_pain_point}'.\n\n"
            f"Foi apresentado a você o seguinte projeto para você assinar/comprar por ${pitch.price_usd_monthly:.0f} USD/mês:\n"
            f"Nome: {pitch.name}\n"
            f"Proposta: {pitch.tagline}\n"
            f"Problema que diz resolver: {pitch.problem_statement}\n"
            f"Solução: {pitch.solution_description}\n\n"
            f"Na sua análise matemática interna, você {'COMPRARIA' if evaluation.would_buy else 'NÃO COMPRARIA'}. "
            f"Sua nota para o produto foi {evaluation.pmf_score}/10.\n\n"
            f"Responda estritamente em formato JSON com estas 4 chaves:\n"
            f'{{"verbatim_quote": "sua opinião sincera em 2 frases em primeira pessoa", '
            f'"top_objection": "a principal barreira ou risco que você vê", '
            f'"key_buying_trigger": "o que exatamente faria você fechar negócio agora", '
            f'"sentiment": "Muito Otimista" ou "Moderadamente Interessado" ou "Cético" ou "Rejeição Imediata"}}\n'
            f"NÃO use markdown, responda apenas o JSON puro."
        )

        for model_name in [DEFAULT_SLM_MODEL, FALLBACK_FAST_MODEL]:
            try:
                res = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.4, "num_predict": 250},
                    },
                    timeout=8.0,
                )
                if res.status_code == 200:
                    data = res.json()
                    raw_text = data.get("response", "").strip()
                    parsed = json.loads(raw_text)
                    return CohortFocusGroupFeedback(
                        archetype=persona.archetype,
                        region=persona.region,
                        representative_name=persona.name,
                        country=persona.country,
                        verbatim_quote=parsed.get("verbatim_quote", ""),
                        top_objection=parsed.get("top_objection", ""),
                        key_buying_trigger=parsed.get("key_buying_trigger", ""),
                        sentiment=parsed.get("sentiment", "Cético"),
                        estimated_wtp_usd=evaluation.max_wtp_usd,
                    )
            except Exception as e:
                logger.debug(f"Falha ao chamar Ollama ({model_name}): {e}")
                continue

        return None

    def _generate_contextual_focus_feedback(
        self,
        persona: Persona,
        evaluation: PersonaEvaluation,
        pitch: ProjectPitch,
    ) -> CohortFocusGroupFeedback:
        """
        Gera feedback qualitativo contextualizado de alta fidelidade sem depender de conexão externa.
        """
        if evaluation.would_buy:
            sentiment = "Muito Otimista" if evaluation.pmf_score >= 8.5 else "Moderadamente Interessado"
            quote = (
                f"Como {persona.archetype.value} aqui em {persona.country}, a proposta de {pitch.name} atende "
                f"diretamente minha necessidade diária. O valor de ${pitch.price_usd_monthly:.0f}/mês está dentro do que "
                f"eu pagaria sem hesitar, desde que a entrega seja tão fluida quanto o pitch promete."
            )
            objection = "Garantia de que a migração inicial não exigirá semanas de setup ou retrabalho."
            trigger = "Oferecer um período de teste sem cartão de crédito com suporte direto no onboarding."
        else:
            sentiment = "Cético" if evaluation.pmf_score >= 5.0 else "Rejeição Imediata"
            quote = (
                f"Achei a ideia de {pitch.name} interessante no papel, mas para a minha realidade em {persona.country}, "
                f"o preço de ${pitch.price_usd_monthly:.0f}/mês é desproporcional ao benefício perceptível. "
                f"Eu prefiro continuar com ferramentas legadas ou scripts próprios por enquanto."
            )
            objection = f"Preço acima da capacidade orçamentária imediata e dúvida se resolve: '{persona.primary_pain_point}'."
            trigger = "Reduzir o preço de entrada ou disponibilizar uma camada freemium/self-hosted."

        return CohortFocusGroupFeedback(
            archetype=persona.archetype,
            region=persona.region,
            representative_name=persona.name,
            country=persona.country,
            verbatim_quote=quote,
            top_objection=objection,
            key_buying_trigger=trigger,
            sentiment=sentiment,
            estimated_wtp_usd=evaluation.max_wtp_usd,
        )
