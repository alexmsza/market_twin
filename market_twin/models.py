"""
Market Twin • Modelos Pydantic de Domínio
Estrutura de dados para simulação estocástica de mercado e focus groups sintéticos.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Region(str, Enum):
    LATAM = "LATAM"
    NORTH_AMERICA = "North America"
    EUROPE = "Europe"
    APAC = "Asia-Pacific"
    MEA = "Middle East & Africa"


class PersonaArchetype(str, Enum):
    ENTERPRISE_BUYER = "Enterprise Director / Buyer"
    SMB_FOUNDER = "SMB Owner / Startup Founder"
    FREELANCER_DEV = "Software Engineer / Tech Freelancer"
    TECH_ENTHUSIAST = "Tech Enthusiast / Early Adopter"
    EVERYDAY_CONSUMER = "Everyday Consumer"
    BUDGET_CONSCIOUS_STUDENT = "Student / Budget-Conscious User"


class ProjectCategory(str, Enum):
    SAAS_B2B = "SaaS B2B"
    CONSUMER_APP = "Consumer App / Mobile"
    FINTECH = "FinTech / Payments"
    DEVTOOL = "Developer Tool / Infra"
    AI_AGENT = "AI Agent / Automation"
    EDTECH = "EdTech / Learning"
    ECOMMERCE = "E-Commerce / Retail"
    HARDWARE_IOT = "Hardware / IoT"


class PricingModel(str, Enum):
    FIXED_SUBSCRIPTION = "Mensalidade Fixa"
    USAGE_BASED = "Pay-per-use / Consumo"
    FREEMIUM = "Freemium"
    ONE_TIME = "Pagamento Único"


class ProjectPitch(BaseModel):
    name: str = Field(..., description="Nome do projeto ou produto")
    tagline: str = Field(..., description="Proposta de valor em uma frase")
    category: ProjectCategory = Field(default=ProjectCategory.SAAS_B2B)
    problem_statement: str = Field(..., description="Dor crítica resolvida pelo projeto")
    solution_description: str = Field(..., description="Como a solução resolve o problema de forma única")
    price_usd_monthly: float = Field(default=29.0, ge=0.0, description="Preço mensal em USD")
    pricing_model: PricingModel = Field(default=PricingModel.FIXED_SUBSCRIPTION)
    target_regions: List[Region] = Field(
        default_factory=lambda: [
            Region.LATAM,
            Region.NORTH_AMERICA,
            Region.EUROPE,
            Region.APAC,
            Region.MEA,
        ]
    )


class Persona(BaseModel):
    id: int
    name: str
    region: Region
    country: str
    archetype: PersonaArchetype
    age: int
    tech_savviness: float = Field(..., ge=0.0, le=1.0)
    monthly_disposable_income_usd: float = Field(..., ge=0.0)
    price_sensitivity: float = Field(..., ge=0.0, le=1.0)
    skepticism_level: float = Field(..., ge=0.0, le=1.0)
    primary_pain_point: str


class PersonaEvaluation(BaseModel):
    persona_id: int
    would_buy: bool
    pmf_score: float = Field(..., ge=0.0, le=10.0)
    nps_category: str  # "promoter", "passive", "detractor"
    max_wtp_usd: float  # Maximum Willingness to Pay
    perceived_value_ratio: float


class CohortFocusGroupFeedback(BaseModel):
    archetype: PersonaArchetype
    region: Region
    representative_name: str
    country: str
    verbatim_quote: str
    top_objection: str
    key_buying_trigger: str
    sentiment: str  # "Muito Otimista", "Moderadamente Interessado", "Cético", "Rejeição Imediata"
    estimated_wtp_usd: float


class PricePointDemand(BaseModel):
    price_usd: float
    conversion_rate: float
    projected_revenue_index: float


class RegionalMetric(BaseModel):
    region: str
    total_population: int
    converted_users: int
    conversion_rate: float
    average_pmf: float
    average_wtp_usd: float


class SimulationResult(BaseModel):
    project_name: str
    population_size: int
    pmf_index: float = Field(..., description="Índice PMF de 0 a 100")
    nps_score: float = Field(..., description="Net Promoter Score de -100 a +100")
    conversion_rate: float = Field(..., description="Taxa de conversão percentual (0 a 100)")
    verdict: str  # "FORTE RECOMENDAÇÃO (GO)", "GO CONDICIONAL", "ALTO RISCO (CAUTION)", "NÃO INVESTIR (NO-GO)"
    verdict_badge: str  # "success", "warning", "danger"
    verdict_rationale: str
    breakdown_by_region: List[RegionalMetric]
    price_elasticity: List[PricePointDemand]
    focus_group: List[CohortFocusGroupFeedback]
    simulation_duration_seconds: float
    timestamp: str
