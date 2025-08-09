"""Database management commands."""

import typer
from rich.console import Console
from typing import Optional
import time

console = Console()
app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """💾 Database management operations."""
    if ctx.invoked_subcommand is None:
        # Run status command by default
        status()


@app.command()
def migrate(
    revision: Optional[str] = typer.Option(
        None, "--revision", "-r", help="Target revision"
    ),
):
    """🔄 Run database migrations."""
    console.print("🔄 [bold blue]Running database migrations...[/bold blue]")

    try:
        from alembic import command
        from alembic.config import Config
        from sejm_whiz.database import get_database_config
        import os

        # Get database configuration
        db_config = get_database_config()

        # Set up Alembic config
        alembic_cfg = Config(
            os.path.join(os.path.dirname(__file__), "../../database/alembic.ini")
        )
        alembic_cfg.set_main_option("sqlalchemy.url", db_config.database_url)

        # Run migration
        if revision:
            command.upgrade(alembic_cfg, revision)
        else:
            command.upgrade(alembic_cfg, "head")

        console.print("✅ [bold green]Database migrations completed![/bold green]")

    except ImportError:
        console.print("❌ [bold red]Database migration is not available.[/bold red]")
        console.print("Missing requirements:")
        console.print("  • Alembic migration framework")
        console.print("  • Database connection configuration")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Migration failed: {str(e)}[/bold red]")
        raise typer.Exit(1)


@app.command()
def seed(
    dataset: str = typer.Option("sample", "--dataset", "-d", help="Dataset to seed"),
):
    """🌱 Seed database with test data."""
    console.print("❌ [bold red]Database seeding is not implemented yet.[/bold red]")
    console.print(
        "This command would populate the database with test data but requires:"
    )
    console.print("  • Test data generation scripts")
    console.print("  • Sample legal documents and metadata")
    console.print("  • User management system")
    console.print("  • Data validation and integrity checks")
    console.print(f"  • Requested dataset: {dataset}")
    raise typer.Exit(1)


@app.command()
def backup(
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Backup file path"
    ),
):
    """💾 Create database backup."""
    console.print("❌ [bold red]Database backup is not implemented yet.[/bold red]")
    console.print("This command would create a database backup but requires:")
    console.print("  • PostgreSQL dump utilities (pg_dump)")
    console.print("  • Database connection and authentication")
    console.print("  • Backup compression and encryption")
    console.print("  • Backup validation and verification")
    backup_file = output or f"sejm-whiz-backup-{time.strftime('%Y%m%d-%H%M%S')}.sql"
    console.print(f"  • Target file: {backup_file}")
    raise typer.Exit(1)


@app.command()
def restore(
    backup_file: str = typer.Argument(..., help="Backup file to restore"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """🔄 Restore database from backup."""
    console.print("❌ [bold red]Database restore is not implemented yet.[/bold red]")
    console.print("This command would restore a database backup but requires:")
    console.print("  • PostgreSQL restore utilities (psql)")
    console.print("  • Database connection and authentication")
    console.print("  • Backup validation and integrity checks")
    console.print("  • Safe restore procedures with rollback")
    console.print(f"  • Source file: {backup_file}")
    console.print(f"  • Confirmation required: {not confirm}")
    raise typer.Exit(1)


@app.command()
def reset(confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")):
    """🗑️ Reset database (WARNING: Deletes all data)."""
    console.print("❌ [bold red]Database reset is not implemented yet.[/bold red]")
    console.print("This command would completely reset the database but requires:")
    console.print("  • Dangerous operation safeguards")
    console.print("  • Database schema recreation scripts")
    console.print("  • Migration system integration")
    console.print("  • Backup creation before reset")
    console.print(f"  • Confirmation required: {not confirm}")
    console.print(
        "⚠️ [bold yellow]This would DELETE ALL DATA - use with extreme caution[/bold yellow]"
    )
    raise typer.Exit(1)


@app.command()
def status():
    """📊 Show database status and statistics."""
    console.print("📊 [bold blue]Database Status[/bold blue]")

    try:
        from rich.table import Table
        from sejm_whiz.database import get_database_config, DatabaseManager

        config = get_database_config()
        db_manager = DatabaseManager(config)

        # Connection info table
        conn_table = Table(title="Connection Information")
        conn_table.add_column("Property", style="cyan")
        conn_table.add_column("Value", style="green")

        conn_table.add_row("Host", config.host)
        conn_table.add_row("Port", str(config.port))
        conn_table.add_row("Database", config.database)

        # Test connection
        if db_manager.test_connection():
            conn_table.add_row("Status", "✅ Connected")
        else:
            conn_table.add_row("Status", "❌ Connection failed")

        console.print(conn_table)

        # Try to get real statistics
        try:
            from sqlalchemy import text

            with db_manager.engine.connect() as conn:
                # Get table statistics from information_schema
                stats_query = text("""
                SELECT
                    table_name,
                    COALESCE(n_tup_ins, 0) as row_count,
                    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size
                FROM information_schema.tables
                LEFT JOIN pg_stat_user_tables ON pg_stat_user_tables.relname = tables.table_name
                WHERE table_schema = 'public'
                ORDER BY table_name;
                """)

                result = conn.execute(stats_query)
                stats_data = result.fetchall()

                if stats_data:
                    stats_table = Table(title="Database Statistics")
                    stats_table.add_column("Table", style="cyan")
                    stats_table.add_column("Rows", style="magenta")
                    stats_table.add_column("Size", style="green")

                    for row in stats_data:
                        stats_table.add_row(row[0], str(row[1]), row[2])

                    console.print(stats_table)
                else:
                    console.print("⚠️ [yellow]No database statistics available[/yellow]")

        except Exception as e:
            console.print(f"⚠️ [yellow]Could not retrieve statistics: {str(e)}[/yellow]")

    except ImportError:
        console.print("❌ [bold red]Database status check not available.[/bold red]")
        console.print("Missing requirements:")
        console.print("  • Database connection components")
        console.print("  • Configuration management")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ [bold red]Status check failed: {str(e)}[/bold red]")
        raise typer.Exit(1)
