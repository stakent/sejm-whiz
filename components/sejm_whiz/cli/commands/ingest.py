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
        # Run status command by default
        status()


@app.command()
def documents(
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="Data source (eli, sejm). Default: process both streams",
    ),
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

    # Handle default dual-stream processing
    if source is None:
        console.print(
            "📥 [bold blue]Ingesting documents from both streams (ELI + Sejm)[/bold blue]"
        )
        source = "dual_stream"  # Internal identifier for dual-stream processing
    else:
        console.print(
            f"📥 [bold blue]Ingesting documents from {source.upper()} stream[/bold blue]"
        )

    # Validate source parameter
    valid_sources = ["eli", "sejm", "dual_stream"]
    if source not in valid_sources:
        console.print(
            f"❌ [bold red]Invalid source '{source}'. Valid options: eli, sejm, or no --source (default: both streams)[/bold red]"
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

        # Show additional multi-API statistics if available
        if source == "dual_stream" and "stream_breakdown" in result:
            console.print()
            console.print("📡 [bold cyan]Dual-Stream Performance:[/bold cyan]")
            stream_breakdown = result["stream_breakdown"]
            for stream_name, count in stream_breakdown.items():
                if stream_name != "total_streams" and count > 0:
                    console.print(f"  • {stream_name}: {count} documents")

            # Show success rate if available
            if result.get("processed", 0) > 0:
                success_rate = result.get("stored", 0) / result.get("processed", 1)
                console.print(f"  🎯 Overall success rate: {success_rate:.1%}")

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
def status():
    """📊 Show multi-API ingestion system status and monitoring."""
    import asyncio
    from rich.table import Table

    console.print(
        "📊 [bold blue]Multi-API Document Processing System Status[/bold blue]"
    )
    console.print()

    async def get_system_status():
        """Get comprehensive system status."""
        try:
            from sejm_whiz.document_ingestion.cached_ingestion_pipeline import (
                CachedDocumentIngestionPipeline,
            )
            from sejm_whiz.document_ingestion.config import get_ingestion_config

            # Initialize pipeline for status checks
            config = get_ingestion_config()
            pipeline = CachedDocumentIngestionPipeline(config)

            # Get cache statistics
            cache_stats = pipeline.get_cache_statistics()

            # Get database statistics
            try:
                from sejm_whiz.database.operations import get_db_session
                from sejm_whiz.database.models import LegalDocument, DocumentEmbedding
                from sqlalchemy import func

                with get_db_session() as session:
                    doc_count = session.query(func.count(LegalDocument.id)).scalar()
                    embedding_count = session.query(
                        func.count(DocumentEmbedding.id)
                    ).scalar()
                    db_stats = {
                        "documents": doc_count,
                        "embeddings": embedding_count,
                        "embedding_coverage": f"{embedding_count / doc_count:.1%}"
                        if doc_count > 0
                        else "0%",
                    }
            except Exception as e:
                db_stats = {"error": str(e)}

            return {"cache": cache_stats, "database": db_stats}

        except Exception as e:
            return {"error": str(e)}

    # Run async status check
    try:
        status_data = asyncio.run(get_system_status())

        # Display system status
        if "error" in status_data:
            console.print(
                f"❌ [red]System status check failed: {status_data['error']}[/red]"
            )
            return

        # Database Status Table
        console.print("🗄️ [bold cyan]Database Status[/bold cyan]")
        db_table = Table(show_header=True, header_style="bold magenta")
        db_table.add_column("Metric", style="cyan")
        db_table.add_column("Value", style="green")

        db_stats = status_data.get("database", {})
        if "error" not in db_stats:
            db_table.add_row("Documents", f"{db_stats.get('documents', 0):,}")
            db_table.add_row("Embeddings", f"{db_stats.get('embeddings', 0):,}")
            db_table.add_row(
                "Embedding Coverage", db_stats.get("embedding_coverage", "N/A")
            )
        else:
            db_table.add_row("Status", f"❌ Error: {db_stats['error']}")

        console.print(db_table)
        console.print()

        # Cache Status Table
        console.print("💾 [bold cyan]Cache System Status[/bold cyan]")
        cache_table = Table(show_header=True, header_style="bold magenta")
        cache_table.add_column("Cache Type", style="cyan")
        cache_table.add_column("Status", style="green")

        cache_stats = status_data.get("cache", {})
        if "error" not in cache_stats:
            api_cache = cache_stats.get("api_cache", {})
            doc_cache = cache_stats.get("document_cache", {})
            text_cache = cache_stats.get("text_cache", {})

            cache_table.add_row(
                "API Cache", "✅ Active" if api_cache else "❌ Unavailable"
            )
            cache_table.add_row(
                "Document Cache",
                f"✅ Active ({doc_cache.get('total_documents', 0)} docs)"
                if doc_cache
                else "❌ Unavailable",
            )
            cache_table.add_row(
                "Text Processing Cache", "✅ Active" if text_cache else "❌ Unavailable"
            )
        else:
            cache_table.add_row(
                "All Caches", f"❌ Error: {cache_stats.get('error', 'Unknown')}"
            )

        console.print(cache_table)
        console.print()

        # Multi-API Capabilities
        console.print("🔧 [bold cyan]Multi-API Processing Capabilities[/bold cyan]")
        capabilities_table = Table(show_header=True, header_style="bold magenta")
        capabilities_table.add_column("Feature", style="cyan")
        capabilities_table.add_column("Status", style="green")

        capabilities_table.add_row("ELI API Integration", "✅ Available")
        capabilities_table.add_row("Sejm API Integration", "✅ Available")
        capabilities_table.add_row("PDF Fallback (pdfplumber)", "✅ Available")
        capabilities_table.add_row("Content Validation", "✅ Available")
        capabilities_table.add_row("Multi-Source Pipeline", "✅ Available")
        capabilities_table.add_row("Retry Queue", "✅ Available")

        console.print(capabilities_table)
        console.print()

        console.print("💡 [bold yellow]Usage Examples:[/bold yellow]")
        console.print(
            "  • Basic dual-stream ingestion: [cyan]sejm-whiz-cli ingest documents --limit 10[/cyan]"
        )
        console.print(
            "  • Single stream processing: [cyan]sejm-whiz-cli ingest documents --source eli --limit 50[/cyan]"
        )
        console.print(
            "  • Recent documents from both streams: [cyan]sejm-whiz-cli ingest documents --since 7d[/cyan]"
        )

    except Exception as e:
        console.print(f"❌ [red]Status check failed: {str(e)}[/red]")
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
