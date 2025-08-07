"""System management commands."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

console = Console()
app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """🔧 System management operations."""
    if ctx.invoked_subcommand is None:
        # Run status command by default
        status(ctx)


@app.command()
def status(ctx: typer.Context):
    """📊 Show comprehensive system health status."""
    console.print("🔍 [bold blue]Checking system components...[/bold blue]")

    # Get current environment from context
    current_env = ctx.obj.get("env", "local") if ctx.obj else "local"

    # Create status table with environment in title
    table = Table(title=f"🏛️ Sejm-Whiz System Status ({current_env})")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")

    # Check services with spinner
    services = [
        ("PostgreSQL", _check_postgres, "Database with pgvector"),
        ("Redis", _check_redis, "Cache and job queue"),
        ("API Server", _check_api, "FastAPI web service"),
        ("Embeddings Model", _check_embeddings, "HerBERT model"),
        ("Search Index", _check_search_index, "Vector similarity"),
    ]

    for service_name, check_func, description in services:
        with console.status(f"Checking {service_name}..."):
            status, details = check_func()
        table.add_row(service_name, status, f"{description} - {details}")

    console.print(table)


@app.command()
def start(
    services: Optional[str] = typer.Option(
        "all", "--services", "-s", help="Services to start (all, api, db, redis)"
    ),
):
    """🚀 Start system services."""
    console.print("❌ [bold red]Service management is not implemented yet.[/bold red]")
    console.print("This command would start system services but requires:")
    console.print("  • Service management integration (systemd, docker-compose)")
    console.print("  • Service configuration and health checks")
    console.print("  • Proper permissions and deployment setup")
    console.print(f"  • Requested services: {services}")
    raise typer.Exit(1)


@app.command()
def stop(
    services: Optional[str] = typer.Option(
        "all", "--services", "-s", help="Services to stop (all, api, db, redis)"
    ),
):
    """🛑 Stop system services."""
    console.print("❌ [bold red]Service management is not implemented yet.[/bold red]")
    console.print("This command would stop system services but requires:")
    console.print("  • Service management integration (systemd, docker-compose)")
    console.print("  • Service process identification and control")
    console.print("  • Graceful shutdown procedures")
    console.print(f"  • Requested services: {services}")
    raise typer.Exit(1)


@app.command()
def restart(
    services: Optional[str] = typer.Option(
        "all", "--services", "-s", help="Services to restart"
    ),
):
    """🔄 Restart system services."""
    console.print("❌ [bold red]Service management is not implemented yet.[/bold red]")
    console.print("This command would restart system services but requires:")
    console.print("  • Service management integration (systemd, docker-compose)")
    console.print("  • Coordinated stop/start procedures")
    console.print("  • Service dependency management")
    console.print(f"  • Requested services: {services}")
    raise typer.Exit(1)


@app.command()
def logs(
    service: Optional[str] = typer.Option(
        "api", "--service", "-s", help="Service to view logs for"
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """📋 View system logs."""
    console.print("❌ [bold red]Log viewing is not implemented yet.[/bold red]")
    console.print("This command would display service logs but requires:")
    console.print("  • Log file location detection")
    console.print("  • Log format parsing and filtering")
    console.print("  • Real-time log streaming for --follow")
    console.print(f"  • Requested service: {service}")
    console.print(f"  • Show {lines} lines, follow: {follow}")
    raise typer.Exit(1)


def _check_postgres():
    """Check PostgreSQL connection."""
    try:
        from sejm_whiz.database import get_database_config, DatabaseManager

        config = get_database_config()
        db_manager = DatabaseManager(config)

        # Test basic connectivity
        if db_manager.test_connection():
            # Check for pgvector extension
            if db_manager.test_pgvector_extension():
                return "✅ Online", "Connected, pgvector loaded"
            else:
                return "⚠️ Partial", "Connected, pgvector missing"
        else:
            return "❌ Offline", "Connection test failed"
    except Exception as e:
        return "❌ Offline", f"Connection failed: {str(e)}"


def _check_redis():
    """Check Redis connection."""
    try:
        from sejm_whiz.redis import check_redis_health, get_redis_client

        # Test Redis connection health
        health_result = check_redis_health()
        if health_result.get("connection") and health_result.get("ping"):
            # Get queue info
            try:
                client = get_redis_client()
                queue_length = client.llen("job_queue") or 0
                return "✅ Online", f"Connected, {queue_length} jobs queued"
            except Exception:
                return "✅ Online", "Connected, queue status unknown"
        else:
            error_msg = health_result.get("error", "Health check failed")
            return "❌ Offline", f"Health check failed: {error_msg}"
    except Exception as e:
        return "❌ Offline", f"Connection failed: {str(e)}"


def _check_api():
    """Check API server."""
    try:
        import httpx
        import os

        # Environment-aware host detection
        cli_env = os.getenv("SEJM_WHIZ_CLI_ENV", "local")

        if cli_env == "p7":
            api_host = "p7"
            api_port = "8001"
        else:
            # Default to environment variables or localhost
            api_host = os.getenv("API_HOST", "localhost")
            api_port = os.getenv("API_PORT", "8001")

        api_url = f"http://{api_host}:{api_port}/health"

        # Test API health endpoint
        with httpx.Client(timeout=5.0) as client:
            response = client.get(api_url)
            if response.status_code == 200:
                return "✅ Online", f"Host {api_host}:{api_port}, health check passed"
            else:
                return (
                    "⚠️ Issues",
                    f"Host {api_host}:{api_port}, HTTP {response.status_code}",
                )

    except Exception as e:
        return "❌ Offline", f"Server not responding at {api_host}:{api_port}: {str(e)}"


def _check_embeddings():
    """Check embeddings model."""
    try:
        from sejm_whiz.embeddings import HerBERTEncoder

        # Try to initialize the model
        encoder = HerBERTEncoder()
        test_text = "Test embedding"
        embeddings = encoder.encode([test_text])

        if embeddings and len(embeddings) > 0:
            return "✅ Ready", f"HerBERT model loaded, dims: {len(embeddings[0])}"
        else:
            return "❌ Error", "Model failed to generate embeddings"

    except ImportError:
        return "❌ Missing", "HerBERT encoder not available"
    except Exception as e:
        return "❌ Error", f"Model error: {str(e)}"


def _check_search_index():
    """Check search index."""
    try:
        from sejm_whiz.database import get_database_config, DatabaseManager
        from sqlalchemy import text

        # Check document count in database
        config = get_database_config()
        db_manager = DatabaseManager(config)

        # Count documents with embeddings
        with db_manager.engine.connect() as conn:
            # Simple query to count documents with embeddings
            query = text(
                "SELECT COUNT(*) FROM legal_documents WHERE embedding IS NOT NULL"
            )
            result = conn.execute(query)
            doc_count = result.scalar() or 0

        if doc_count > 0:
            return "✅ Ready", f"{doc_count:,} documents indexed"
        else:
            return "⚠️ Empty", "No documents indexed yet"

    except Exception as e:
        return "❌ Error", f"Index check failed: {str(e)}"
