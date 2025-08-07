"""Data ingestion management commands."""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
import time

console = Console()
app = typer.Typer()


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
    console.print(
        f"📥 [bold blue]Ingesting documents from {source.upper()} API[/bold blue]"
    )

    # Handle date parameters
    from datetime import datetime, timedelta
    import re

    # Process date range
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

    # Simulate document ingestion with progress
    total_docs = limit or 5000

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Phase 1: Fetch documents
        fetch_task = progress.add_task("🔍 Fetching document list...", total=None)
        time.sleep(2)
        progress.update(
            fetch_task, description="✅ Document list fetched", completed=True
        )

        # Phase 2: Process documents
        process_task = progress.add_task(
            f"📝 Processing {total_docs:,} documents", total=total_docs
        )

        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            progress.update(
                process_task,
                description=f"📝 Processing batch {i//batch_size + 1} ({i+1}-{batch_end})",
                advance=batch_size,
            )
            time.sleep(0.1)

    console.print(
        f"✅ [bold green]Ingested {total_docs:,} documents successfully![/bold green]"
    )


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
    console.print(f"🧠 [bold blue]Generating embeddings with {model}[/bold blue]")
    console.print(f"  🖥️ Device: {'GPU' if gpu else 'CPU'}")
    console.print(f"  📦 Batch size: {batch_size}")

    # Check for unprocessed documents
    pending_docs = 1250  # Simulated count

    if pending_docs == 0:
        console.print(
            "✅ [bold green]All documents already have embeddings![/bold green]"
        )
        return

    with Progress() as progress:
        task = progress.add_task(
            f"🧠 Generating embeddings for {pending_docs} documents", total=pending_docs
        )

        for i in range(0, pending_docs, batch_size):
            time.sleep(0.2)  # Simulate processing time
            progress.update(task, advance=min(batch_size, pending_docs - i))

    console.print("✅ [bold green]Embeddings generated successfully![/bold green]")


@app.command()
def schedule(
    interval: str = typer.Option("daily", "--interval", "-i", help="Schedule interval"),
    source: str = typer.Option("eli", "--source", "-s", help="Data source"),
    time_str: str = typer.Option("02:00", "--time", "-t", help="Schedule time (HH:MM)"),
):
    """⏰ Schedule automatic data ingestion jobs."""
    console.print(f"⏰ [bold blue]Scheduling {source.upper()} ingestion[/bold blue]")
    console.print(f"  📅 Interval: {interval}")
    console.print(f"  🕐 Time: {time_str}")

    # Simulate job scheduling
    time.sleep(1)

    job_id = f"{source}-ingestion-{interval}"
    console.print(f"✅ [bold green]Scheduled job '{job_id}' successfully![/bold green]")
    console.print(f"  📋 Next run: Tomorrow at {time_str}")


@app.command()
def status():
    """📊 Show ingestion pipeline status."""
    from rich.table import Table

    console.print("📊 [bold blue]Ingestion Pipeline Status[/bold blue]")

    # Jobs table
    jobs_table = Table(title="Scheduled Jobs")
    jobs_table.add_column("Job ID", style="cyan")
    jobs_table.add_column("Source", style="magenta")
    jobs_table.add_column("Schedule", style="green")
    jobs_table.add_column("Last Run", style="yellow")
    jobs_table.add_column("Status", style="blue")

    jobs_table.add_row(
        "eli-daily", "ELI API", "Daily 02:00", "2025-08-06 02:00", "✅ Success"
    )
    jobs_table.add_row(
        "sejm-weekly", "Sejm API", "Weekly Sun 03:00", "2025-08-04 03:00", "✅ Success"
    )
    jobs_table.add_row(
        "embeddings-daily", "Local", "Daily 04:00", "2025-08-06 04:00", "✅ Success"
    )

    console.print(jobs_table)

    # Statistics table
    stats_table = Table(title="Ingestion Statistics (Last 30 Days)")
    stats_table.add_column("Source", style="cyan")
    stats_table.add_column("Documents", style="magenta")
    stats_table.add_column("Success Rate", style="green")
    stats_table.add_column("Avg Duration", style="yellow")

    stats_table.add_row("ELI API", "1,245", "98.5%", "12m 34s")
    stats_table.add_row("Sejm API", "892", "99.2%", "8m 15s")
    stats_table.add_row("Embeddings", "2,137", "99.8%", "45m 22s")

    console.print(stats_table)


@app.command()
def logs(
    job_id: Optional[str] = typer.Option(None, "--job", "-j", help="Specific job ID"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """📋 View ingestion job logs."""
    if job_id:
        console.print(f"📋 [bold blue]Ingestion logs for job '{job_id}'[/bold blue]")
    else:
        console.print("📋 [bold blue]Recent ingestion logs[/bold blue]")

    if follow:
        console.print("Following logs... Press Ctrl+C to stop")

    # Simulate log output
    sample_logs = [
        "2025-08-06 02:00:01 [INFO] Starting ELI document ingestion",
        "2025-08-06 02:00:05 [INFO] Fetched 1,234 document URLs",
        "2025-08-06 02:02:30 [INFO] Processed batch 1/13 (100 documents)",
        "2025-08-06 02:05:45 [INFO] Processed batch 2/13 (100 documents)",
        "2025-08-06 02:12:15 [INFO] All documents ingested successfully",
        "2025-08-06 02:12:16 [INFO] Ingestion completed in 12m 15s",
    ]

    for log_line in sample_logs[-lines:]:
        console.print(log_line)
