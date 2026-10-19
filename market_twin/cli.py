"""
Market Twin • Interface de Linha de Comando (CLI)
Executa simulações estocásticas de até 100.000 clones virtuais diretamente no terminal.
"""

from __future__ import annotations

import argparse
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from market_twin.models import ProjectCategory, ProjectPitch, PricingModel
from market_twin.report_generator import ExecutiveReportGenerator
from market_twin.simulator import MarketTwinSimulator

console = Console(force_terminal=True, soft_wrap=True)



def run_cli(args: argparse.Namespace) -> int:
    console.print(
        Panel.fit(
            "[bold cyan]MARKET TWIN[/bold cyan] • [bold white]Simulador de População Sintética & PMF Engine[/bold white]\n"
            f"[dim]População: {args.population:,} personas | Preço: ${args.price:.2f}/mês | Categoria: {args.category}[/dim]",
            border_style="cyan",
        )
    )

    pitch = ProjectPitch(
        name=args.name or "Projeto Sem Nome",
        tagline=args.tagline or args.pitch,
        category=ProjectCategory(args.category),
        problem_statement=args.problem or args.pitch,
        solution_description=args.solution or f"Solução completa para: {args.pitch}",
        price_usd_monthly=float(args.price),
        pricing_model=PricingModel.FIXED_SUBSCRIPTION,
    )

    simulator = MarketTwinSimulator(seed=args.seed)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task1 = progress.add_task(
            f"[cyan]Sintetizando {args.population:,} clones virtuais e calculando micro-econometria...",
            total=None,
        )
        result = simulator.run_simulation(
            pitch=pitch,
            population_size=args.population,
            use_llm_focus_group=not args.fast,
        )
        progress.update(task1, completed=100)

    # 1. Exibir Veredito Executivo
    badge_style = "bold green" if result.verdict_badge == "success" else (
        "bold yellow" if result.verdict_badge == "warning" else "bold red"
    )
    console.print(
        Panel(
            f"[{badge_style}]VEREDITO DE INVESTIMENTO: {result.verdict}[/{badge_style}]\n\n"
            f"{result.verdict_rationale}\n\n"
            f"[dim]Inferência concluída em {result.simulation_duration_seconds}s para {result.population_size:,} indivíduos.[/dim]",
            title="[bold white]Avaliação Executiva de Investimento[/bold white]",
            border_style="green" if result.verdict_badge == "success" else ("yellow" if result.verdict_badge == "warning" else "red"),
        )
    )

    # 2. Tabela de Métricas Principais
    kpi_table = Table(title="Indicadores-Chave de Aderência (PMF & Demanda)", header_style="bold magenta")
    kpi_table.add_column("Métrica", style="cyan", no_wrap=True)
    kpi_table.add_column("Valor Simulado", style="bold white")
    kpi_table.add_column("Benchmark de Mercado / Diagnóstico", style="dim")

    pmf_diag = "Tração Forte (>60)" if result.pmf_index >= 60 else ("Moderada (45-59)" if result.pmf_index >= 45 else "Crítica (<45)")
    kpi_table.add_row("Índice PMF (0-100)", f"{result.pmf_index:.1f} / 100", pmf_diag)

    nps_diag = "Excelente (>+20)" if result.nps_score >= 20 else ("Neutro (0 a +19)" if result.nps_score >= 0 else "Risco Detratores (<0)")
    kpi_table.add_row("Net Promoter Score (NPS)", f"{result.nps_score:+.1f}", nps_diag)

    cr_diag = "Excelente conversão (>15%)" if result.conversion_rate >= 15 else ("Média de mercado (8-14%)" if result.conversion_rate >= 8 else "Baixa conversão (<8%)")
    kpi_table.add_row("Taxa de Conversão Global", f"{result.conversion_rate:.2f}%", cr_diag)
    console.print(kpi_table)

    # 3. Tabela por Macrorregião
    reg_table = Table(title="Desempenho por Macrorregião Global", header_style="bold blue")
    reg_table.add_column("Região", style="bold white")
    reg_table.add_column("Amostra", justify="right")
    reg_table.add_column("Compradores", justify="right", style="green")
    reg_table.add_column("Conversão", justify="right", style="bold yellow")
    reg_table.add_column("Média PMF", justify="right")
    reg_table.add_column("WTP Médio (USD)", justify="right", style="bold cyan")

    for reg in result.breakdown_by_region:
        reg_table.add_row(
            reg.region,
            f"{reg.total_population:,}",
            f"{reg.converted_users:,}",
            f"{reg.conversion_rate:.2f}%",
            f"{reg.average_pmf:.2f}/10",
            f"${reg.average_wtp_usd:.2f}",
        )
    console.print(reg_table)

    # 4. Tabela de Elasticidade de Preço
    price_table = Table(title="Curva de Elasticidade de Preço (Van Westendorp)", header_style="bold green")
    price_table.add_column("Preço Simulado (USD/mês)", style="bold cyan")
    price_table.add_column("Conversão Estimada", justify="right")
    price_table.add_column("Índice de Faturamento Relativo", justify="right", style="bold magenta")

    for pe in result.price_elasticity:
        price_table.add_row(
            f"${pe.price_usd:.2f}",
            f"{pe.conversion_rate:.2f}%",
            f"{pe.projected_revenue_index:.1f}%",
        )
    console.print(price_table)

    # 5. Voz do Cliente: Focus Group (SLM Tier 2)
    console.print("\n[bold yellow]Focus Group Qualitativo (Depoimentos Representativos):[/bold yellow]")
    for fg in result.focus_group:
        console.print(
            Panel(
                f"[italic]\"{fg.verbatim_quote}\"[/italic]\n\n"
                f"[bold red]Objeção Central:[/bold red] {fg.top_objection}\n"
                f"[bold green]Gatilho de Compra:[/bold green] {fg.key_buying_trigger}\n"
                f"[dim]Disposição a Pagar Máxima Estimada: ${fg.estimated_wtp_usd:.2f} USD[/dim]",
                title=f"[bold white]{fg.representative_name}[/bold white] ({fg.country} • {fg.region.value}) — [cyan]{fg.archetype.value}[/cyan] [bold]({fg.sentiment})[/bold]",
                border_style="cyan",
            )
        )

    # 6. Exportação em PDF se solicitado
    if args.pdf:
        pdf_gen = ExecutiveReportGenerator()
        pdf_gen.generate_pdf(result, output_path=args.pdf)
        console.print(f"\n[bold green]✔ Dossiê Executivo PDF gerado com sucesso em:[/bold green] [bold underline]{args.pdf}[/bold underline]")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="market-twin",
        description="Market Twin • Simulador de População Sintética para Validação de PMF e Investimentos",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Executa simulação de mercado para um projeto")
    run_parser.add_argument("--pitch", type=str, required=True, help="Descrição ou pitch do projeto")
    run_parser.add_argument("--name", type=str, default="Meu Projeto", help="Nome do produto")
    run_parser.add_argument("--tagline", type=str, default="", help="Tagline do produto")
    run_parser.add_argument("--problem", type=str, default="", help="Dor principal resolvida")
    run_parser.add_argument("--solution", type=str, default="", help="Solução proposta")
    run_parser.add_argument("--price", type=float, default=29.0, help="Preço mensal em USD")
    run_parser.add_argument(
        "--category",
        type=str,
        default="SaaS B2B",
        choices=[c.value for c in ProjectCategory],
        help="Categoria de negócio do projeto",
    )
    run_parser.add_argument("--population", type=int, default=10000, help="Tamanho da população sintética (ex: 10000)")
    run_parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade Monte Carlo")
    run_parser.add_argument("--fast", action="store_true", help="Pular chamadas lentas ao Ollama e usar síntese determinística")
    run_parser.add_argument("--pdf", type=str, default="", help="Caminho para exportar o relatório em PDF (ex: relatorio.pdf)")

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(run_cli(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
