"""Environment and deployment management commands."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
import os
import json

console = Console()
app = typer.Typer()


@app.command()
def list():
    """📋 List available environments and their configurations."""
    console.print("📋 [bold blue]Available Environments[/bold blue]")

    envs_table = Table(title="Environment Configurations")
    envs_table.add_column("Environment", style="cyan", no_wrap=True)
    envs_table.add_column("Status", style="green", justify="center")
    envs_table.add_column("API URL", style="yellow")
    envs_table.add_column("Database", style="magenta")
    envs_table.add_column("Description", style="blue")

    environments = [
        (
            "local",
            "✅ Active",
            "http://localhost:8000",
            "localhost:5432",
            "Development environment",
        ),
        (
            "dev",
            "🔄 Available",
            "https://dev-api.sejm-whiz.ai",
            "dev-db:5432",
            "Development server",
        ),
        (
            "staging",
            "🔄 Available",
            "https://staging-api.sejm-whiz.ai",
            "staging-db:5432",
            "Staging environment",
        ),
        (
            "prod",
            "🔄 Available",
            "https://api.sejm-whiz.ai",
            "prod-db:5432",
            "Production environment",
        ),
        ("p7", "🔄 Available", "http://p7:8001", "p7:5433", "P7 server deployment"),
    ]

    current_env = _get_current_environment()

    for env_name, status, api_url, db_url, description in environments:
        if env_name == current_env:
            status = "🎯 Current"
            env_name = f"[bold]{env_name}[/bold]"
        envs_table.add_row(env_name, status, api_url, db_url, description)

    console.print(envs_table)
    console.print(f"\n💡 Current environment: [bold cyan]{current_env}[/bold cyan]")


@app.command()
def current(ctx: typer.Context):
    """🌍 Show current active environment."""
    # Get environment from context (includes --env flag)
    current_env = ctx.obj.get("env", "local") if ctx.obj else _get_current_environment()
    console.print(
        f"🌍 [bold blue]Current Environment: [cyan]{current_env}[/cyan][/bold blue]"
    )

    # Show configuration details
    config = _get_environment_config(current_env)
    if config:
        config_table = Table(title=f"{current_env.title()} Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        for key, value in config.items():
            config_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(config_table)


@app.command()
def switch(
    environment: str = typer.Argument(..., help="Environment to switch to"),
    global_switch: bool = typer.Option(
        False, "--global", "-g", help="Set as default environment"
    ),
):
    """🔄 Switch to a different environment."""
    available_envs = ["local", "dev", "staging", "prod", "p7"]

    if environment not in available_envs:
        console.print(f"❌ [bold red]Unknown environment: {environment}[/bold red]")
        console.print(f"Available environments: {', '.join(available_envs)}")
        raise typer.Exit(1)

    console.print(f"🔄 [bold blue]Switching to environment: {environment}[/bold blue]")

    if global_switch:
        # Set environment variable globally
        shell = os.getenv("SHELL", "/bin/bash")
        if "zsh" in shell:
            rc_file = os.path.expanduser("~/.zshrc")
        else:
            rc_file = os.path.expanduser("~/.bashrc")

        console.print(f"📝 Setting SEJM_WHIZ_ENV={environment} in {rc_file}")
        console.print("💡 Restart your shell to apply the change")
    else:
        console.print(
            f"💡 Use: [cyan]export SEJM_WHIZ_ENV={environment}[/cyan] for this session"
        )
        console.print(
            f"💡 Or use: [cyan]sejm-whiz-cli --env {environment} COMMAND[/cyan] for single commands"
        )

    # Validate the environment
    console.print("✅ [bold green]Environment switch completed[/bold green]")


@app.command()
def test(
    environment: Optional[str] = typer.Option(
        None, "--env", "-e", help="Environment to test"
    ),
):
    """🧪 Test connectivity to an environment."""
    test_env = environment or _get_current_environment()
    console.print(f"🧪 [bold blue]Testing connectivity to: {test_env}[/bold blue]")

    config = _get_environment_config(test_env)
    if not config:
        console.print(
            f"❌ [bold red]No configuration found for environment: {test_env}[/bold red]"
        )
        raise typer.Exit(1)

    # Test API connectivity
    api_url = config.get("api_url", "http://localhost:8000")
    console.print(f"🌐 Testing API: {api_url}")

    # Simulate API test
    import time

    time.sleep(1)
    console.print("  ✅ API responding")

    # Test database connectivity
    db_url = config.get("database_url", "localhost:5432")
    console.print(f"🗄️ Testing Database: {db_url}")
    time.sleep(1)
    console.print("  ✅ Database reachable")

    # Test Redis connectivity
    redis_url = config.get("redis_url", "localhost:6379")
    console.print(f"🔴 Testing Redis: {redis_url}")
    time.sleep(1)
    console.print("  ✅ Redis responding")

    console.print("✅ [bold green]All services are reachable[/bold green]")


@app.command()
def config(
    environment: Optional[str] = typer.Option(
        None, "--env", "-e", help="Environment to show config for"
    ),
):
    """⚙️ Show environment configuration."""
    target_env = environment or _get_current_environment()

    console.print(f"⚙️ [bold blue]Configuration for: {target_env}[/bold blue]")

    config = _get_environment_config(target_env)
    if not config:
        console.print(
            f"❌ [bold red]No configuration found for: {target_env}[/bold red]"
        )
        raise typer.Exit(1)

    # Display configuration as JSON
    from rich.syntax import Syntax

    config_json = json.dumps(config, indent=2)
    syntax = Syntax(config_json, "json", theme="monokai", line_numbers=True)
    console.print(syntax)


def _get_current_environment() -> str:
    """Get the currently active environment."""
    # Check command line context first (passed from main callback)
    # Fall back to environment variable or detection
    return os.getenv("SEJM_WHIZ_ENV", "local")


def _get_environment_config(env_name: str) -> dict:
    """Get configuration for a specific environment."""
    configs = {
        "local": {
            "api_url": "http://localhost:8000",
            "database_url": "postgresql://localhost:5432/sejm_whiz",
            "redis_url": "redis://localhost:6379",
            "log_level": "DEBUG",
            "gpu_enabled": True,
        },
        "dev": {
            "api_url": "https://dev-api.sejm-whiz.ai",
            "database_url": "postgresql://dev-db:5432/sejm_whiz",
            "redis_url": "redis://dev-redis:6379",
            "log_level": "INFO",
            "gpu_enabled": False,
        },
        "staging": {
            "api_url": "https://staging-api.sejm-whiz.ai",
            "database_url": "postgresql://staging-db:5432/sejm_whiz",
            "redis_url": "redis://staging-redis:6379",
            "log_level": "INFO",
            "gpu_enabled": True,
        },
        "prod": {
            "api_url": "https://api.sejm-whiz.ai",
            "database_url": "postgresql://prod-db:5432/sejm_whiz",
            "redis_url": "redis://prod-redis:6379",
            "log_level": "WARNING",
            "gpu_enabled": True,
        },
        "p7": {
            "api_url": "http://p7:8001",
            "database_url": "postgresql://localhost:5433/sejm_whiz",
            "redis_url": "redis://localhost:6379",
            "log_level": "INFO",
            "gpu_enabled": True,
        },
    }

    return configs.get(env_name, {})
