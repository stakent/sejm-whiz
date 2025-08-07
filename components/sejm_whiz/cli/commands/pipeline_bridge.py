"""Bridge between CLI commands and backend pipeline components."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

# Import existing pipeline components
from sejm_whiz.document_ingestion.cached_ingestion_pipeline import (
    CachedDocumentIngestionPipeline,
)
from sejm_whiz.document_ingestion.config import get_ingestion_config


class CliPipelineOrchestrator:
    """Orchestrates pipeline execution with CLI-appropriate progress reporting."""

    def __init__(self, console: Console):
        self.console = console
        self.logger = logging.getLogger(__name__)

        # Statistics tracking
        self.stats = {
            "processed": 0,
            "stored": 0,
            "skipped": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
        }

    async def execute_ingestion(
        self,
        source: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        batch_size: int = 100,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Execute document ingestion with progress reporting."""
        self.stats["start_time"] = datetime.now()

        try:
            if source in ["eli", "both"]:
                eli_result = await self._execute_eli_ingestion(
                    start_date, end_date, limit, batch_size, force
                )
                self._merge_stats(eli_result)

            if source in ["sejm", "both"]:
                sejm_result = await self._execute_sejm_ingestion(
                    start_date, end_date, limit, batch_size, force
                )
                self._merge_stats(sejm_result)

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

        finally:
            self.stats["end_time"] = datetime.now()
            duration = self.stats["end_time"] - self.stats["start_time"]
            self.stats["duration"] = str(duration).split(".")[0]  # Remove microseconds

        return self.stats

    async def _execute_eli_ingestion(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
        batch_size: int,
        force: bool,
    ) -> Dict[str, Any]:
        """Execute ELI document ingestion using DocumentIngestionPipeline."""
        self.console.print(
            "🔄 [bold cyan]Starting ELI document ingestion...[/bold cyan]"
        )

        try:
            # Initialize cached pipeline for better performance
            config = get_ingestion_config()
            config.batch_size = batch_size
            # Note: force parameter not directly supported by current config

            pipeline = CachedDocumentIngestionPipeline(config)

            # Calculate days for recent documents if date range is specified
            days = 7  # Default
            if start_date and end_date:
                days = (end_date - start_date).days
            elif start_date:
                days = (datetime.now() - start_date).days

            # Execute with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                # Create progress task
                task = progress.add_task(
                    f"📄 Ingesting ELI documents (last {days} days)", total=None
                )

                # Execute pipeline
                result = await pipeline.ingest_recent_documents(days=days)

                # Update progress as completed
                progress.update(
                    task, description="✅ ELI ingestion completed", completed=True
                )

                self.console.print(
                    f"✅ [green]ELI ingestion completed: {result.get('documents_stored', 0)} documents[/green]"
                )

                # Add cache statistics to result
                try:
                    cache_stats = pipeline.get_cache_statistics()
                    result["cache_statistics"] = cache_stats

                    # Show cache performance info
                    api_cache = cache_stats.get("api_cache", {})
                    if api_cache.get("total_files", 0) > 0:
                        self.console.print(
                            f"💾 [dim]Cache: {api_cache.get('total_files', 0)} files, "
                            f"{api_cache.get('total_size_mb', 0):.1f}MB[/dim]"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to get cache statistics: {e}")

                return result

        except Exception as e:
            self.console.print(f"❌ [red]ELI ingestion failed: {str(e)}[/red]")
            raise

    async def _execute_sejm_ingestion(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
        batch_size: int,
        force: bool,
    ) -> Dict[str, Any]:
        """Execute Sejm document ingestion using enhanced data processor."""
        self.console.print(
            "🔄 [bold cyan]Starting Sejm document ingestion...[/bold cyan]"
        )

        try:
            # Import enhanced data processor components
            import sys
            import os

            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

            from enhanced_data_processor import ComprehensiveSejmDataStep
            from sejm_whiz.data_pipeline.core import DataPipeline

            # Execute with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                # Create progress task
                task = progress.add_task("🏛️ Ingesting Sejm documents", total=None)

                # Create pipeline with Sejm step
                pipeline = DataPipeline("cli_sejm_ingestion")
                sejm_step = ComprehensiveSejmDataStep()
                pipeline.add_step(sejm_step)

                # Prepare data with date filtering
                input_data = {}
                if limit:
                    input_data["limit"] = limit

                # Execute pipeline
                result = await pipeline.process(input_data)

                # Update progress as completed
                progress.update(
                    task, description="✅ Sejm ingestion completed", completed=True
                )

                # Extract statistics from result
                sejm_stats = {
                    "processed": len(result.get("proceedings", []))
                    + len(result.get("votings", []))
                    + len(result.get("interpellations", [])),
                    "stored": result.get("documents_stored", 0),
                    "skipped": result.get("documents_skipped", 0),
                    "failed": result.get("documents_failed", 0),
                }

                self.console.print(
                    f"✅ [green]Sejm ingestion completed: {sejm_stats['processed']} documents[/green]"
                )

                return sejm_stats

        except ImportError as e:
            self.console.print(
                f"❌ [red]Sejm pipeline components not available: {str(e)}[/red]"
            )
            # Return empty stats for graceful degradation
            return {"processed": 0, "stored": 0, "skipped": 0, "failed": 0}
        except Exception as e:
            self.console.print(f"❌ [red]Sejm ingestion failed: {str(e)}[/red]")
            raise

    def _merge_stats(self, pipeline_result: Dict[str, Any]) -> None:
        """Merge pipeline results into overall statistics."""
        self.stats["processed"] += pipeline_result.get("processed", 0)
        self.stats["stored"] += pipeline_result.get("stored", 0) or pipeline_result.get(
            "documents_stored", 0
        )
        self.stats["skipped"] += pipeline_result.get(
            "skipped", 0
        ) or pipeline_result.get("documents_skipped", 0)
        self.stats["failed"] += pipeline_result.get("failed", 0) or pipeline_result.get(
            "documents_failed", 0
        )
