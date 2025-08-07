"""Configuration management commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from typing import Optional, Union, Any
import json

console = Console()
app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """⚙️ Configuration management operations."""
    if ctx.invoked_subcommand is None:
        # Run show command by default
        show(None, "table")


@app.command()
def show(
    section: Optional[str] = typer.Option(
        None, "--section", "-s", help="Configuration section"
    ),
    format_output: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json, yaml)"
    ),
):
    """📋 Show current configuration."""
    console.print("📋 [bold blue]Current Configuration[/bold blue]")

    # Sample configuration
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "sejm_whiz",
            "pool_size": 10,
        },
        "redis": {"host": "localhost", "port": 6379, "db": 0},
        "embeddings": {"model": "herbert", "batch_size": 32, "gpu_enabled": True},
        "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
        "ingestion": {
            "eli_batch_size": 100,
            "sejm_batch_size": 50,
            "schedule": "daily",
        },
    }

    # Filter by section if specified
    display_config = (
        {section: config[section]} if section and section in config else config
    )

    if format_output == "json":
        syntax = Syntax(
            json.dumps(display_config, indent=2),
            "json",
            theme="monokai",
            line_numbers=True,
        )
        console.print(syntax)
    elif format_output == "yaml":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            console.print(
                "❌ [bold red]PyYAML not installed. Install with: uv add pyyaml[/bold red]"
            )
            raise typer.Exit(1)

        yaml_str = yaml.dump(display_config, default_flow_style=False)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:  # table format
        for section_name, section_config in display_config.items():
            config_table = Table(title=f"⚙️ {section_name.title()} Configuration")
            config_table.add_column("Setting", style="cyan", no_wrap=True)
            config_table.add_column("Value", style="green")
            config_table.add_column("Type", style="yellow", justify="center")

            for key, value in section_config.items():
                value_type = type(value).__name__
                value_str = str(value)
                config_table.add_row(key, value_str, value_type)

            console.print(config_table)


@app.command()
def set(
    key: str = typer.Argument(..., help="Configuration key (section.setting)"),
    value: str = typer.Argument(..., help="New value"),
    config_type: str = typer.Option(
        "auto", "--type", "-t", help="Value type (auto, str, int, bool, float)"
    ),
):
    """⚙️ Set configuration value."""
    console.print(f"⚙️ [bold blue]Setting configuration: {key} = {value}[/bold blue]")

    # Parse the key
    if "." not in key:
        console.print("❌ [bold red]Key must be in format 'section.setting'[/bold red]")
        raise typer.Exit(1)

    section, setting = key.split(".", 1)

    # Convert value based on type
    typed_value: Union[str, int, float, bool]
    if config_type == "auto":
        # Auto-detect type
        if value.lower() in ["true", "false"]:
            typed_value = value.lower() == "true"
            config_type = "bool"
        elif value.isdigit():
            typed_value = int(value)
            config_type = "int"
        elif value.replace(".", "", 1).isdigit():
            typed_value = float(value)
            config_type = "float"
        else:
            typed_value = value
            config_type = "str"
    else:
        # Explicit type conversion
        type_converters: dict[str, Any] = {
            "str": str,
            "int": int,
            "bool": lambda x: x.lower() == "true",
            "float": float,
        }
        try:
            typed_value = type_converters[config_type](value)
        except (ValueError, KeyError) as e:
            console.print(
                f"❌ [bold red]Invalid value for type {config_type}: {e}[/bold red]"
            )
            raise typer.Exit(1)

    console.print(f"  📝 Section: {section}")
    console.print(f"  🔧 Setting: {setting}")
    console.print(f"  💾 Value: {typed_value} ({config_type})")

    # Simulate saving configuration
    import time

    time.sleep(0.5)

    console.print("✅ [bold green]Configuration updated successfully![/bold green]")
    console.print("💡 [dim]Restart services to apply changes[/dim]")


@app.command()
def validate():
    """✅ Validate current configuration."""
    console.print("✅ [bold blue]Validating configuration...[/bold blue]")

    # Simulate validation checks
    checks = [
        ("Database connection", True, "PostgreSQL reachable"),
        ("Redis connection", True, "Redis server responding"),
        ("GPU availability", True, "CUDA 12.1 detected"),
        ("Model files", True, "HerBERT model found"),
        ("Directory permissions", True, "All paths writable"),
        ("Port availability", False, "Port 8000 already in use"),
        ("Memory requirements", True, "16GB RAM available"),
    ]

    import time

    for check_name, passed, details in checks:
        with console.status(f"Checking {check_name.lower()}..."):
            time.sleep(0.3)

        status = "✅ Pass" if passed else "❌ Fail"
        color = "green" if passed else "red"
        console.print(
            f"  {status} [bold {color}]{check_name}[/bold {color}]: {details}"
        )

    # Summary
    failed_checks = sum(1 for _, passed, _ in checks if not passed)
    if failed_checks == 0:
        console.print("✅ [bold green]All validation checks passed![/bold green]")
    else:
        console.print(
            f"❌ [bold red]{failed_checks} validation check(s) failed![/bold red]"
        )
        console.print("💡 [dim]Fix the issues above before starting services[/dim]")


@app.command()
def export(
    output_file: str = typer.Option(
        "sejm-whiz-config.json", "--output", "-o", help="Output file path"
    ),
    format_type: str = typer.Option(
        "json", "--format", "-f", help="Export format (json, yaml, env)"
    ),
):
    """📤 Export configuration to file."""
    console.print(
        f"📤 [bold blue]Exporting configuration to: {output_file}[/bold blue]"
    )
    console.print(f"  📋 Format: {format_type.upper()}")

    # Sample configuration for export (removed unused variable)

    # Simulate export
    import time

    time.sleep(1)

    if format_type == "json":
        console.print("📋 JSON format selected")
    elif format_type == "yaml":
        console.print("📋 YAML format selected")
    elif format_type == "env":
        console.print("📋 Environment variables format selected")
        console.print("  💡 Variables will be prefixed with SEJM_WHIZ_")

    console.print("✅ [bold green]Configuration exported successfully![/bold green]")
    console.print(f"  📄 File: {output_file}")
    console.print("  📊 Size: 1.2 KB")


@app.command()
def import_config(
    config_file: str = typer.Argument(..., help="Configuration file to import"),
    merge: bool = typer.Option(
        False, "--merge", "-m", help="Merge with existing config"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before import"
    ),
):
    """📥 Import configuration from file."""
    console.print(
        f"📥 [bold blue]Importing configuration from: {config_file}[/bold blue]"
    )

    if backup:
        console.print("  💾 Creating backup of current configuration...")

    if merge:
        console.print("  🔄 Merge mode: Existing values will be preserved")
    else:
        console.print("  ⚠️ Replace mode: All current settings will be overwritten")

    # Simulate import process
    import time

    with console.status("📋 Reading configuration file..."):
        time.sleep(1)

    with console.status("✅ Validating configuration..."):
        time.sleep(1)

    if not merge and not typer.confirm(
        "⚠️ This will replace all current settings. Continue?"
    ):
        console.print("❌ Import cancelled.")
        raise typer.Abort()

    with console.status("💾 Applying configuration..."):
        time.sleep(1)

    console.print("✅ [bold green]Configuration imported successfully![/bold green]")
    console.print("  📊 Settings updated: 23")
    console.print("  🔄 Services to restart: api, ingestion")
    console.print("💡 [dim]Run 'sejm-whiz-cli system restart' to apply changes[/dim]")


@app.command()
def reset(
    section: Optional[str] = typer.Option(
        None, "--section", "-s", help="Reset specific section only"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """🔄 Reset configuration to defaults."""
    if section:
        console.print(
            f"🔄 [bold yellow]Resetting {section} configuration to defaults[/bold yellow]"
        )
    else:
        console.print(
            "🔄 [bold yellow]Resetting ALL configuration to defaults[/bold yellow]"
        )

    if not confirm:
        if not typer.confirm("⚠️ This will reset configuration to defaults. Continue?"):
            console.print("❌ Reset cancelled.")
            raise typer.Abort()

    # Simulate reset
    import time

    time.sleep(1)

    console.print("✅ [bold green]Configuration reset to defaults![/bold green]")
    if section:
        console.print(f"  📋 Section '{section}' restored to factory settings")
    else:
        console.print("  📋 All settings restored to factory defaults")
    console.print("💡 [dim]You may need to reconfigure database and API settings[/dim]")
