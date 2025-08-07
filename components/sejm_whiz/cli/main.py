"""Main CLI application for sejm-whiz system management."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from sejm_whiz import __version__
from .commands import system, database, ingest, search, model, config, dev, env

# Initialize Rich console for beautiful output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="sejm-whiz-cli",
    help="🏛️ Polish Legal Document Analysis System",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=False,
)

# Add subcommands
app.add_typer(system.app, name="system", help="🔧 System management operations")
app.add_typer(database.app, name="db", help="🗄️ Database operations")
app.add_typer(ingest.app, name="ingest", help="📥 Data ingestion management")
app.add_typer(search.app, name="search", help="🔍 Search operations")
app.add_typer(model.app, name="model", help="🤖 ML model management")
app.add_typer(config.app, name="config", help="⚙️ Configuration management")
app.add_typer(env.app, name="env", help="🌍 Environment management")
app.add_typer(dev.app, name="dev", help="🛠️ Development tools")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", help="Show version and exit"
    ),
    env: Optional[str] = typer.Option(
        None, "--env", "-e", help="Target environment (local, dev, staging, prod, p7)"
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Configuration profile to use"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Custom configuration file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    🏛️ **Sejm-Whiz**: Polish Legal Document Analysis System

    A comprehensive CLI for managing the sejm-whiz AI system that analyzes
    Polish legal documents and predicts legislative changes.

    **Quick Start:**
    ```
    # Check system status
    uv run python sejm-whiz-cli.py system status

    # Ingest documents
    uv run python sejm-whiz-cli.py ingest documents --source eli

    # Search for documents
    uv run python sejm-whiz-cli.py search query "ustawa o ochronie danych"

    # Or create alias for shorter commands:
    alias sejm-whiz-cli="uv run python ./sejm-whiz-cli.py"
    ```
    """
    if version:
        console.print(
            f"[bold blue]sejm-whiz-cli[/bold blue] version [green]{__version__}[/green]"
        )
        raise typer.Exit()

    # Store global context for subcommands
    ctx.ensure_object(dict)
    current_env = env or _detect_environment()
    ctx.obj["env"] = current_env
    ctx.obj["profile"] = profile
    ctx.obj["config_file"] = config_file
    ctx.obj["verbose"] = verbose

    # Set environment variable for configuration modules to pick up
    import os

    os.environ["SEJM_WHIZ_CLI_ENV"] = current_env

    # Always show environment info (unless showing version or help only)
    if ctx.invoked_subcommand is not None:
        current_env = ctx.obj["env"]
        env_color = _get_environment_color(current_env)
        console.print(f"🌍 [{env_color}]Environment: {current_env}[/{env_color}]")
        if verbose:
            if profile:
                console.print(f"📋 [dim]Profile: {profile}[/dim]")
            if config_file:
                console.print(f"⚙️ [dim]Config file: {config_file}[/dim]")
        console.print()  # Add spacing

    # If no subcommand is invoked, show help
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _detect_environment() -> str:
    """Detect current environment from various sources."""
    import os

    # Environment variable
    env_var = os.getenv("SEJM_WHIZ_ENV")
    if env_var:
        return env_var

    # Check for deployment-specific indicators
    if os.path.exists("/root/tmp/sejm-whiz"):
        return "p7"
    elif os.path.exists("/app"):
        return "docker"
    elif os.getenv("KUBERNETES_SERVICE_HOST"):
        return "k8s"
    else:
        return "local"


def _get_environment_color(env: str) -> str:
    """Get color for environment display."""
    color_map = {
        "local": "cyan",
        "dev": "blue",
        "staging": "yellow",
        "prod": "red",
        "p7": "magenta",
        "docker": "blue",
        "k8s": "green",
    }
    return color_map.get(env, "white")


@app.command()
def help():
    """📚 Show comprehensive CLI help and examples."""
    console.print("📚 [bold blue]Sejm-Whiz CLI Help & Quick Reference[/bold blue]")
    console.print()

    # Quick start section
    console.print("🚀 [bold green]Quick Start[/bold green]")
    console.print("```")
    console.print("# Check system status")
    console.print("uv run python sejm-whiz-cli.py system status")
    console.print()
    console.print("# Ingest recent documents")
    console.print(
        "uv run python sejm-whiz-cli.py ingest documents --since 7d --source eli"
    )
    console.print()
    console.print("# Search for documents")
    console.print(
        'uv run python sejm-whiz-cli.py search query "ustawa o ochronie danych"'
    )
    console.print("```")
    console.print()

    # Common commands
    console.print("💡 [bold green]Most Used Commands[/bold green]")
    common_table = Table()
    common_table.add_column("Command", style="cyan", no_wrap=True)
    common_table.add_column("Description", style="green")
    common_table.add_column("Example", style="yellow")

    common_commands = [
        (
            "system status",
            "Check system health",
            "uv run python sejm-whiz-cli.py system status",
        ),
        (
            "ingest documents",
            "Import legal documents",
            "uv run python sejm-whiz-cli.py ingest documents --since 30d",
        ),
        (
            "search query",
            "Search documents",
            'uv run python sejm-whiz-cli.py search query "RODO"',
        ),
        ("db migrate", "Update database", "uv run python sejm-whiz-cli.py db migrate"),
        ("config show", "View settings", "uv run python sejm-whiz-cli.py config show"),
    ]

    for cmd, desc, example in common_commands:
        common_table.add_row(cmd, desc, example)

    console.print(common_table)
    console.print()

    # Date parameters help
    console.print("📅 [bold green]Date Filtering (for ingest commands)[/bold green]")
    date_table = Table()
    date_table.add_column("Parameter", style="cyan")
    date_table.add_column("Format", style="magenta")
    date_table.add_column("Example", style="yellow")
    date_table.add_column("Description", style="green")

    date_params = [
        ("--from", "YYYY-MM-DD", "--from 2025-01-13", "Start from specific date"),
        ("--to", "YYYY-MM-DD", "--to 2025-02-01", "End at specific date"),
        ("--since", "Xd/Xw/Xm/Xy", "--since 30d", "Relative time from now"),
    ]

    for param, format_str, example, desc in date_params:
        date_table.add_row(param, format_str, example, desc)

    console.print(date_table)
    console.print()
    console.print(
        "📝 [dim]Relative date examples: 1d, 7d, 30d, 1w, 2w, 1m, 6m, 1y[/dim]"
    )
    console.print()

    # Help resources
    console.print("📖 [bold green]Getting More Help[/bold green]")
    console.print(
        "• Detailed command help: [cyan]uv run python sejm-whiz-cli.py COMMAND --help[/cyan]"
    )
    console.print(
        "• Subcommand help: [cyan]uv run python sejm-whiz-cli.py COMMAND SUBCOMMAND --help[/cyan]"
    )
    console.print("• Complete guide: [cyan]CLI_README.md[/cyan] in project root")
    console.print("• Shell completion: [cyan]./scripts/install-completion.sh[/cyan]")
    console.print()

    console.print("💡 [bold yellow]Pro Tips[/bold yellow]")
    console.print(
        "• Use tab completion for faster command entry (after setting up alias)"
    )
    console.print("• Add [cyan]--verbose[/cyan] to any command for debug output")
    console.print("• Check [cyan]system status[/cyan] before running operations")
    console.print(
        "• Use [cyan]--dry-run[/cyan] flags when available to preview actions"
    )
    console.print(
        "• Create alias: [cyan]alias sejm-whiz-cli='uv run python ./sejm-whiz-cli.py'[/cyan]"
    )


@app.command()
def info():
    """📊 Show system information and component status."""
    table = Table(title="🏛️ Sejm-Whiz System Information")

    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Description", style="green")

    components = [
        ("Core System", "✅ Active", f"Version {__version__}"),
        ("Database", "🔄 Checking...", "PostgreSQL + pgvector"),
        ("Redis Cache", "🔄 Checking...", "Cache and job queue"),
        ("Embeddings", "✅ Ready", "HerBERT Polish model"),
        ("Web API", "🔄 Checking...", "FastAPI server"),
        ("Search Index", "🔄 Checking...", "Vector similarity search"),
    ]

    for component, status, description in components:
        table.add_row(component, status, description)

    console.print(table)
    console.print(
        "\n💡 [bold]Tip:[/bold] Use [cyan]sejm-whiz-cli system status[/cyan] for detailed health check"
    )


if __name__ == "__main__":
    app()
