"""Data ingestion management commands."""

import typer
from rich.console import Console
from typing import Optional

console = Console()
app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """📥 Data ingestion management operations."""
    if ctx.invoked_subcommand is None:
        # Run status command by default if it exists, otherwise show help
        try:
            status()
        except NameError:
            # If status command doesn't exist, show basic info
            console.print("📥 [bold blue]Data Ingestion Management[/bold blue]")
            console.print("Available commands:")
            console.print("  • documents - Ingest legal documents")
            console.print("  • schedule - Manage ingestion schedules")
            console.print("  • status - Check ingestion system status")
            console.print("\nRun with --help to see all options.")


@app.command()
def documents(
    source: str = typer.Option("eli", "--source", "-s", help="Data source (eli, sejm)"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Max documents to ingest"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="Batch processing size"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-ingestion of existing docs"
    ),
    date_from: Optional[str] = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD)"
    ),
    date_to: Optional[str] = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD, default: today)"
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Relative date (1d, 1w, 1m, 1y)"
    ),
):
    """📥 Ingest legal documents from external APIs."""
    import asyncio
    from datetime import datetime, timedelta
    import re

    console.print(
        f"📥 [bold blue]Ingesting documents from {source.upper()} API[/bold blue]"
    )

    # Validate source parameter
    valid_sources = ["eli", "sejm", "both"]
    if source not in valid_sources:
        console.print(
            f"❌ [bold red]Invalid source '{source}'. Valid options: {', '.join(valid_sources)}[/bold red]"
        )
        raise typer.Exit(1)

    # Process date range parameters
    start_date = None
    end_date = datetime.now()

    if since:
        # Parse relative date (1d, 1w, 1m, 1y)
        match = re.match(r"^(\d+)([dwmy])$", since.lower())
        if match:
            amount, unit = int(match.group(1)), match.group(2)
            if unit == "d":
                start_date = end_date - timedelta(days=amount)
            elif unit == "w":
                start_date = end_date - timedelta(weeks=amount)
            elif unit == "m":
                start_date = end_date - timedelta(days=amount * 30)  # Approximate
            elif unit == "y":
                start_date = end_date - timedelta(days=amount * 365)  # Approximate
            if start_date:
                console.print(
                    f"  📅 Since: {since} ago ({start_date.strftime('%Y-%m-%d')})"
                )
        else:
            console.print(
                "❌ [bold red]Invalid --since format. Use: 1d, 1w, 1m, 1y[/bold red]"
            )
            raise typer.Exit(1)

    if date_from:
        try:
            start_date = datetime.strptime(date_from, "%Y-%m-%d")
            console.print(f"  📅 From: {date_from}")
        except ValueError:
            console.print(
                "❌ [bold red]Invalid --from date format. Use: YYYY-MM-DD[/bold red]"
            )
            raise typer.Exit(1)

    if date_to:
        try:
            end_date = datetime.strptime(date_to, "%Y-%m-%d")
            console.print(f"  📅 To: {date_to}")
        except ValueError:
            console.print(
                "❌ [bold red]Invalid --to date format. Use: YYYY-MM-DD[/bold red]"
            )
            raise typer.Exit(1)
    else:
        console.print(f"  📅 To: {end_date.strftime('%Y-%m-%d')} (today)")

    # Validate date range
    if start_date and end_date and start_date > end_date:
        console.print("❌ [bold red]Start date cannot be after end date[/bold red]")
        raise typer.Exit(1)

    if limit:
        console.print(f"  📊 Limit: {limit:,} documents")
    console.print(f"  📦 Batch size: {batch_size}")
    console.print(f"  🔄 Force re-ingestion: {'Yes' if force else 'No'}")

    # Calculate estimated date range
    if start_date:
        date_range_days = (end_date - start_date).days
        console.print(f"  📊 Date range: {date_range_days} days")

    console.print()

    # Execute ingestion pipeline
    try:
        from .pipeline_bridge import CliPipelineOrchestrator

        orchestrator = CliPipelineOrchestrator(console)
        result = asyncio.run(
            orchestrator.execute_ingestion(
                source=source,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                batch_size=batch_size,
                force=force,
            )
        )

        # Display final results
        console.print()
        console.print("✅ [bold green]Ingestion completed successfully![/bold green]")
        console.print(f"  📄 Documents processed: {result.get('processed', 0):,}")
        console.print(f"  💾 Documents stored: {result.get('stored', 0):,}")
        console.print(f"  ⏭️ Documents skipped: {result.get('skipped', 0):,}")
        console.print(f"  ❌ Documents failed: {result.get('failed', 0):,}")
        console.print(f"  ⏱️ Duration: {result.get('duration', 'unknown')}")

    except ImportError:
        console.print("❌ [bold red]Pipeline bridge not found. Creating...[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Ingestion failed: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def embeddings(
    model: str = typer.Option(
        "herbert", "--model", "-m", help="Embedding model to use"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Processing batch size"
    ),
    gpu: bool = typer.Option(True, "--gpu/--cpu", help="Use GPU acceleration"),
):
    """🧠 Generate document embeddings."""
    console.print(
        "❌ [bold red]Embedding generation is not yet implemented.[/bold red]"
    )
    console.print("This command will:")
    console.print("  • Load HerBERT model for Polish text")
    console.print("  • Generate embeddings for ingested documents")
    console.print("  • Store embeddings in PostgreSQL with pgvector")
    console.print("  • Support GPU/CPU processing modes")
    raise typer.Exit(1)


@app.command()
def schedule(
    interval: str = typer.Option("daily", "--interval", "-i", help="Schedule interval"),
    source: str = typer.Option("eli", "--source", "-s", help="Data source"),
    time_str: str = typer.Option("02:00", "--time", "-t", help="Schedule time (HH:MM)"),
):
    """⏰ Schedule automatic data ingestion jobs."""
    console.print("❌ [bold red]Job scheduling is not yet implemented.[/bold red]")
    console.print("This command will:")
    console.print("  • Set up cron-like job scheduling")
    console.print("  • Configure automatic ingestion intervals")
    console.print("  • Manage recurring data updates")
    console.print("  • Send notifications on job completion")
    raise typer.Exit(1)


@app.command()
def status():
    """📊 Show ingestion pipeline status."""
    console.print("❌ [bold red]Status reporting is not yet implemented.[/bold red]")
    console.print("This command will show:")
    console.print("  • Scheduled job status and history")
    console.print("  • Document ingestion statistics")
    console.print("  • Pipeline health metrics")
    console.print("  • Recent error logs and alerts")
    raise typer.Exit(1)


@app.command()
def logs(
    job_id: Optional[str] = typer.Option(None, "--job", "-j", help="Specific job ID"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """📋 View ingestion job logs."""
    console.print("❌ [bold red]Log viewing is not yet implemented.[/bold red]")
    console.print("This command will:")
    console.print("  • Display ingestion job logs")
    console.print("  • Filter logs by job ID or date range")
    console.print("  • Follow live log streams")
    console.print("  • Export logs to files for analysis")
    raise typer.Exit(1)
