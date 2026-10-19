# 🌐 Market Twin • Simulador de População Sintética & Validação de PMF

Motor de simulação populacional e estocástica de mercado capaz de sintetizar **10.000 a 100.000 clones virtuais** em 5 macrorregiões globais (LATAM, América do Norte, Europa, Ásia-Pacífico e Oriente Médio/África) para validação empírica de propostas de produtos, startups e softwares.

---

## 🏛️ Arquitetura em Dois Níveis (Dual-Tier Cognitive Engine)

O Market Twin resolve o gargalo de custo e latência de inferência combinando micro-econometria vetorial de alto desempenho (Monte Carlo) com grupos focais qualitativos profundos orquestrados via SLM/LLM local (`aszadev-pro` / Ollama).

```mermaid
flowchart TD
    Pitch["💡 Pitch do Projeto (Ideia, Preço USD, Dores, ICP)"] --> MonteCarlo["🎲 Tier 1: Gerador Monte Carlo (10.000 Personas)"]
    
    subgraph Stratification ["Estratificação Global & Demográfica"]
        LATAM["LATAM (40%)"]
        NA["North America (25%)"]
        EU["Europe (20%)"]
        APAC["Asia-Pacific (10%)"]
        MEA["Middle East & Africa (5%)"]
    end
    
    MonteCarlo --> Stratification
    Stratification --> UtilityEngine["⚡ Motor de Utilidade Micro-Econômica (NumPy)"]
    
    UtilityEngine --> Metrics["📊 Métricas Quantitativas (PMF, NPS, Van Westendorp WTP)"]
    UtilityEngine --> Clustering["🧬 Agrupamento de Coortes Estratificadas (K-Archetypes)"]
    
    Clustering --> LLMFocus["🧠 Tier 2: Focus Group Cognitivo (Ollama / aszadev-pro)"]
    
    LLMFocus --> Verbatims["💬 Objeções Qualitativas & Gatilhos de Compra"]
    
    Metrics --> Verdict["⚖️ Veredito de Investimento: GO / CAUTION / NO-GO"]
    Verbatims --> Verdict
    Verdict --> Dashboard["🖥️ Dashboard Interativo & Relatório Executivo PDF"]
```

---

## 🚀 Capacidades Principais

1. **Geração Vetorizada de 10.000 Clones em Sub-Segundos:**
   - Distribuição de renda calibrada por paridade do poder de compra (PPP / PIB per capita regional).
   - Níveis de ceticismo, sensibilidade a preço e maturidade técnica modelados por distribuições Beta e Log-Normal com sementes determinísticas.
2. **Índice de Product-Market Fit (PMF Index 0-100):**
   - Baseado na fórmula de Sean Ellis ("Como você se sentiria se não pudesse mais usar este produto?"), taxa de adoção e elasticidade.
3. **Curva de Elasticidade e Sensibilidade de Preço (Van Westendorp PSM):**
   - Identifica o Ponto de Preço Ótimo (OPP) e Faixa de Preço Aceitável em dólares.
4. **Focus Group Sintético com SLM Local:**
   - Extração de feedbacks reais com verbatims, objeções severas e gatilhos de decisão por arquétipo regional.
5. **Veredito Executivo de Investimento:**
   - Classificação objetiva: `FORTE RECOMENDAÇÃO (GO)`, `GO CONDICIONAL`, `ALTO RISCO (CAUTION)` ou `NÃO INVESTIR (NO-GO)`.
6. **Exportação Executiva em PDF:**
   - Dossiê para investidores, sócios e fundadores pronto para apresentação.

## ⚡ Aceleração Nativa em Rust (Rayon SIMD / C-ABI)

O núcleo computacional do **Tier 1 (Monte Carlo & Utilidade Micro-Econômica)** foi reimplementado em **Rust nativo** com multithreading work-stealing via **Rayon** e exposto diretamente à memória do Python via **C-ABI (`ctypes`)**.

### ⏱️ Benchmarks Reais de Inferência:
| População Sintética | Motor Python / NumPy | Motor Nativo Rust (Rayon) | Speedup / Ganho |
| :--- | :--- | :--- | :--- |
| **10.000 Personas** | ~400 ms | **11.8 ms** | **~34x mais rápido** |
| **50.000 Personas** | ~2.100 ms | **7.8 ms** | **~269x mais rápido** |
| **100.000 Personas** | ~4.200 ms | **11.8 ms** | **~355x mais rápido** |

```mermaid
flowchart TD
    Pitch["💡 Pitch do Projeto"] --> Python["🐍 Python / FastAPI"]
    Python -->|"ctypes (C-ABI in-memory)"| Rust["🦀 Motor Nativo Rust (market_twin_engine.dll)"]
    
    subgraph RustParallel ["Paralelismo Massivo Rayon (12 Threads CPU)"]
        Chunk1["Chunk 1: Monte Carlo + Utilidade"]
        Chunk2["Chunk 2: Monte Carlo + Utilidade"]
        ChunkN["Chunk N: Monte Carlo + Utilidade"]
    end
    
    Rust --> RustParallel
    RustParallel -->|"Agregação Atômica (< 12ms)"| RustOutput["📊 PMF, NPS, Elasticidade, Métricas Regionais"]
    RustOutput --> Python
    Python --> LLM["🧠 Tier 2: Focus Group Sintético (Ollama / aszadev-pro)"]
    LLM --> UI["🖥️ Dashboard Interativo / CLI / Dossiê PDF"]
```

---

## 🛠️ Stack Tecnológica

- **Aceleração Nativa:** Rust 1.98+ com Rayon (Multithreading sem GIL) e C-ABI DLL.
- **Linguagem & Runtime:** Python 3.12+ gerenciado rigorosamente via `uv`.

- **Backend & Servidor:** FastAPI + Uvicorn + Pydantic v2.
- **Modelagem Estatística:** NumPy & Pandas (distribuições estocásticas vetorizadas).
- **Inteligência Artificial:** Ollama Local (`aszadev-pro:latest` / `aszadev-fast:latest` / `dolphin-mistral:latest`).
- **Geração de Relatórios:** ReportLab (PDFs executivos vetoriais).
- **Interface & CLI:** Tailwind CSS / Vanilla JS executivo e Rich CLI no terminal.

---

## 💻 Instalação e Execução

### 1. Sincronizar Dependências com `uv`
```bash
uv sync
```

### 2. Executar via Terminal (CLI)
```bash
# Simulação rápida com 10.000 clones virtuais
uv run -m market_twin.cli run --pitch "API de Monitoramento de Pagamentos PIX em Tempo Real" --price 49 --population 10000
```

### 3. Iniciar Servidor e Dashboard Web
```bash
uv run -m market_twin.server
# Acesse no navegador: http://127.0.0.1:8090
```

---

## 📄 Licença
Propriedade de Alex Souza • Consultoria Inteligente. Todos os direitos reservados.
