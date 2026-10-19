"""
Market Twin • Gerador Vetorizado de População Sintética (Monte Carlo)
Gera de 1.000 a 100.000 personas virtuais calibradas estatisticamente por macrorregião global.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional
import numpy as np

from market_twin.models import (
    Persona,
    PersonaArchetype,
    Region,
)

# Calibração Demográfica Global
REGION_WEIGHTS = {
    Region.LATAM: 0.35,
    Region.NORTH_AMERICA: 0.25,
    Region.EUROPE: 0.20,
    Region.APAC: 0.15,
    Region.MEA: 0.05,
}

COUNTRIES_BY_REGION = {
    Region.LATAM: ["Brasil", "México", "Colômbia", "Argentina", "Chile", "Peru"],
    Region.NORTH_AMERICA: ["Estados Unidos", "Canadá"],
    Region.EUROPE: ["Alemanha", "Reino Unido", "França", "Espanha", "Holanda", "Portugal"],
    Region.APAC: ["Singapura", "Japão", "Austrália", "Índia", "Coreia do Sul"],
    Region.MEA: ["Emirados Árabes", "África do Sul", "Arábia Saudita", "Nigéria", "Israel"],
}

ARCHETYPE_WEIGHTS = {
    PersonaArchetype.ENTERPRISE_BUYER: 0.10,
    PersonaArchetype.SMB_FOUNDER: 0.20,
    PersonaArchetype.FREELANCER_DEV: 0.25,
    PersonaArchetype.TECH_ENTHUSIAST: 0.15,
    PersonaArchetype.EVERYDAY_CONSUMER: 0.20,
    PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: 0.10,
}

# Renda média disponível mensal por região (parâmetros de log-normal: mu, sigma)
INCOME_PARAMS_BY_REGION = {
    Region.NORTH_AMERICA: (8.4, 0.55),  # mediana ~ $4.400
    Region.EUROPE: (8.0, 0.50),         # mediana ~ $3.000
    Region.APAC: (7.4, 0.75),           # mediana ~ $1.600
    Region.LATAM: (6.8, 0.65),          # mediana ~ $900
    Region.MEA: (6.7, 0.85),            # mediana ~ $800
}

PAIN_POINTS_BY_ARCHETYPE = {
    PersonaArchetype.ENTERPRISE_BUYER: [
        "Falta de compliance, SLA rígido e suporte 24/7",
        "Dificuldade de integração com legado (ERP/Active Directory)",
        "Risco de segurança da informação e vazamento de dados",
        "Custo oculto de migração e treinamento de equipes",
    ],
    PersonaArchetype.SMB_FOUNDER: [
        "Custo operacional elevado com margem apertada",
        "Pouco tempo para configurar ferramentas complexas",
        "Necessidade imediata de ROI comprovado em menos de 30 dias",
        "Atendimento lento dos grandes players de tecnologia",
    ],
    PersonaArchetype.FREELANCER_DEV: [
        "Documentação técnica fraca ou APIs instáveis",
        "Lock-in proprietário sem possibilidade de exportação",
        "Falta de suporte a CLI/SDKs modernos e webhooks",
        "Preços abusivos por assento que penalizam pequenas equipes",
    ],
    PersonaArchetype.TECH_ENTHUSIAST: [
        "Interfaces lentas, engessadas ou com UX dos anos 2000",
        "Falta de recursos de ponta como IA nativa e automações",
        "Lentidão para lançar novas features solicitadas pela comunidade",
    ],
    PersonaArchetype.EVERYDAY_CONSUMER: [
        "Aplicativos com curva de aprendizado confusa",
        "Cobranças automáticas surpresa no cartão de crédito",
        "Falta de suporte humanizado e resolução rápida de problemas",
    ],
    PersonaArchetype.BUDGET_CONSCIOUS_STUDENT: [
        "Ausência de plano gratuito generoso ou desconto de estudante",
        "Exigência de cartão de crédito internacional para testar",
        "Conteúdo ou interface não traduzidos para a língua materna",
    ],
}

FIRST_NAMES = [
    "Alex", "Lucas", "Mariana", "Gabriel", "Beatriz", "Carlos", "Camila", "Rodrigo",
    "John", "Sarah", "Michael", "Emma", "David", "Jessica", "James", "Emily",
    "Pierre", "Sophie", "Hans", "Elena", "Mateo", "Clara", "Marco", "Anna",
    "Kenji", "Yuki", "Aarav", "Priya", "Min-ho", "Ji-won", "Wei", "Mei",
    "Tariq", "Fatima", "Kwame", "Amina", "Zainab", "Omar", "Thabo", "Lindiwe"
]

LAST_NAMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
    "Müller", "Schmidt", "Dubois", "Martin", "García", "Fernández",
    "Tanaka", "Sato", "Sharma", "Patel", "Kim", "Lee", "Chen", "Wang",
    "Al-Mansoor", "Hassan", "Mensah", "Diallo", "Van Der Merwe", "Ndlovu"
]


class SyntheticPopulationGenerator:
    """
    Gerador estocástico vetorizado de alta performance para até 100.000 personas sintéticas.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_population(self, size: int = 10000) -> List[Persona]:
        """
        Gera uma população estocástica de N personas virtuais distribuídas globalmente.
        """
        regions_list = list(REGION_WEIGHTS.keys())
        region_probs = list(REGION_WEIGHTS.values())

        archetypes_list = list(ARCHETYPE_WEIGHTS.keys())
        archetype_probs = list(ARCHETYPE_WEIGHTS.values())

        # Sorteio vetorizado de índices de regiões e arquétipos para preservar os Enums
        region_indices = self.rng.choice(len(regions_list), size=size, p=region_probs)
        archetype_indices = self.rng.choice(len(archetypes_list), size=size, p=archetype_probs)

        # Vetorização de idades (Gaussiana truncada entre 18 e 72)
        raw_ages = self.rng.normal(loc=36, scale=11, size=size)
        ages = np.clip(raw_ages, 18, 72).astype(int)

        # Vetorização de Tech Savviness (Beta(2.5, 2.5) centrado em 0.5)
        tech_savviness = self.rng.beta(2.5, 2.5, size=size)

        # Ajuste de tech savviness por arquétipo
        for i in range(size):
            arch = archetypes_list[archetype_indices[i]]
            if arch in (PersonaArchetype.FREELANCER_DEV, PersonaArchetype.TECH_ENTHUSIAST):
                tech_savviness[i] = min(1.0, tech_savviness[i] * 1.35 + 0.15)
            elif arch == PersonaArchetype.ENTERPRISE_BUYER:
                tech_savviness[i] = min(1.0, tech_savviness[i] * 1.1 + 0.05)
            elif arch == PersonaArchetype.BUDGET_CONSCIOUS_STUDENT:
                tech_savviness[i] = min(1.0, tech_savviness[i] * 1.15)

        # Vetorização de Ceticismo (Beta(3, 2.5) centrado em 0.55)
        skepticism = self.rng.beta(3.0, 2.5, size=size)
        for i in range(size):
            if archetypes_list[archetype_indices[i]] == PersonaArchetype.ENTERPRISE_BUYER:
                skepticism[i] = min(1.0, skepticism[i] * 1.25 + 0.1)

        personas: List[Persona] = []

        # Gerar valores de renda e montar objetos
        for i in range(size):
            reg = regions_list[region_indices[i]]
            arch = archetypes_list[archetype_indices[i]]
            
            # Cálculo de renda por distribuição log-normal calibrada
            mu, sigma = INCOME_PARAMS_BY_REGION[reg]
            income = float(np.exp(self.rng.normal(mu, sigma)))

            # Ajuste de renda pelo arquétipo (Enterprise e SMB têm maior orçamento)
            if arch == PersonaArchetype.ENTERPRISE_BUYER:
                income *= float(self.rng.uniform(3.5, 7.0))
            elif arch == PersonaArchetype.SMB_FOUNDER:
                income *= float(self.rng.uniform(2.0, 4.0))
            elif arch == PersonaArchetype.BUDGET_CONSCIOUS_STUDENT:
                income *= float(self.rng.uniform(0.3, 0.6))

            # Sensibilidade a preço: inversamente proporcional ao log da renda
            # e com dispersão estocástica
            norm_income = np.clip(income / 10000.0, 0.05, 1.0)
            base_sensitivity = 1.0 - (np.log1p(norm_income) / np.log1p(1.0)) * 0.7
            sensitivity = float(np.clip(base_sensitivity + self.rng.normal(0, 0.08), 0.05, 0.98))

            # Sorteio do país da região
            country = str(self.rng.choice(COUNTRIES_BY_REGION[reg]))

            # Dor principal do arquétipo
            pain_point = str(self.rng.choice(PAIN_POINTS_BY_ARCHETYPE[arch]))

            # Nome randômico determinístico
            f_name = FIRST_NAMES[(i + int(self.seed or 0)) % len(FIRST_NAMES)]
            l_name = LAST_NAMES[(i * 3 + int(self.seed or 0)) % len(LAST_NAMES)]
            full_name = f"{f_name} {l_name}"

            personas.append(
                Persona.model_construct(
                    id=i + 1,
                    name=full_name,
                    region=reg,
                    country=country,
                    archetype=arch,
                    age=int(ages[i]),
                    tech_savviness=round(float(tech_savviness[i]), 3),
                    monthly_disposable_income_usd=round(income, 2),
                    price_sensitivity=round(sensitivity, 3),
                    skepticism_level=round(float(skepticism[i]), 3),
                    primary_pain_point=pain_point,
                )
            )

        return personas
