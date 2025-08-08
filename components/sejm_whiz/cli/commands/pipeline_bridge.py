"""Bridge between CLI commands and backend pipeline components."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
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
    StreamType,
)


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

            elif source in ["sejm", "both"]:
                sejm_result = await self._execute_sejm_ingestion(
                    start_date, end_date, limit, batch_size, force
                )
                self._merge_stats(sejm_result)

            elif source == "dual_stream":
                # Process both ELI and Sejm streams simultaneously (default behavior)
                await self._execute_dual_stream_ingestion(
                    start_date, end_date, limit, batch_size, force
                )

            else:
                raise ValueError(
                    f"Invalid source '{source}'. Valid options: eli, sejm, dual_stream"
                )

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
        """Execute ELI document ingestion using DualApiDocumentProcessor."""
        self.console.print(
            "🔄 [bold cyan]Starting ELI document ingestion...[/bold cyan]"
        )

        try:
            # Use CachedDocumentIngestionPipeline with manual database storage for complete workflow
            pipeline = CachedDocumentIngestionPipeline()

            # Get document IDs from ELI API (simplified for CLI usage)
            document_ids = await self._get_eli_document_ids(start_date, end_date, limit)

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
                    f"📄 Processing {len(document_ids)} ELI documents",
                    total=len(document_ids),
                )

                # Process documents individually and store to database
                processed = 0
                stored = 0
                skipped = 0
                failed = 0

                for i, doc_id in enumerate(document_ids):
                    try:
                        # Process document content
                        result = await pipeline.process_document_by_stream(
                            doc_id, StreamType.ELI.value
                        )
                        processed += 1

                        # If processing successful, check for duplicates then store to database
                        if (
                            result.get("content_extracted")
                            and result.get("metadata_extracted")
                            and result.get("act_text")
                        ):
                            try:
                                # Check if document already exists in database
                                existing_doc = (
                                    pipeline.db_operations.get_document_by_eli_id(
                                        doc_id
                                    )
                                )
                                if existing_doc:
                                    skipped += 1
                                else:
                                    # Store processed document to database using DocumentOperations.create_document
                                    doc_id_uuid = (
                                        pipeline.db_operations.create_document(
                                            title=result.get("metadata", {}).get(
                                                "title", f"ELI Document {doc_id}"
                                            ),
                                            content=result.get("act_text", ""),
                                            document_type="eli_document",
                                            eli_identifier=doc_id,
                                            source_url=result.get("source_used", ""),
                                            legal_act_type=result.get(
                                                "metadata", {}
                                            ).get("document_type", "unknown"),
                                            legal_domain="general",
                                        )
                                    )
                                    if doc_id_uuid:
                                        stored += 1
                                    else:
                                        failed += 1
                            except Exception as db_error:
                                self.logger.warning(
                                    f"Database operation failed for {doc_id}: {db_error}"
                                )
                                failed += 1
                        else:
                            failed += 1

                        progress.update(task, advance=1)

                    except Exception as e:
                        failed += 1
                        self.logger.warning(f"Failed to process {doc_id}: {e}")
                        progress.update(task, advance=1)

                # Update progress as completed
                progress.update(
                    task, description="✅ ELI ingestion completed", completed=True
                )

                self.console.print(
                    f"✅ [green]ELI ingestion completed: {stored} documents stored[/green]"
                )

                return {
                    "processed": processed,
                    "stored": stored,
                    "skipped": skipped,
                    "failed": failed,
                    "documents_stored": stored,
                    "documents_failed": failed,
                    "documents_skipped": skipped,
                }

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
        """Execute Sejm document ingestion using DualApiDocumentProcessor."""
        self.console.print(
            "🔄 [bold cyan]Starting Sejm document ingestion...[/bold cyan]"
        )

        try:
            # Use dual approach: DualApiDocumentProcessor + DocumentIngestionPipeline for database operations
            from sejm_whiz.document_ingestion.ingestion_pipeline import (
                DocumentIngestionPipeline,
            )
            from sejm_whiz.document_ingestion.dual_stream_pipeline import (
                DualApiDocumentProcessor,
            )
            from sejm_whiz.sejm_api.client import SejmApiClient
            from sejm_whiz.eli_api.content_validator import BasicContentValidator

            # Initialize components
            sejm_client = SejmApiClient()
            content_validator = BasicContentValidator()
            processor = DualApiDocumentProcessor(
                sejm_client=sejm_client, content_validator=content_validator
            )
            pipeline = DocumentIngestionPipeline()

            # Get document IDs from Sejm API (simplified for CLI usage)
            document_ids = await self._get_sejm_document_ids(
                start_date, end_date, limit
            )

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
                    f"🏛️ Processing {len(document_ids)} Sejm documents",
                    total=len(document_ids),
                )

                # Process documents individually with dual-API processor + manual storage
                processed = 0
                stored = 0
                skipped = 0
                failed = 0

                for i, doc_id in enumerate(document_ids):
                    try:
                        # Process document content using DualApiDocumentProcessor
                        result = await processor.process_sejm_document(doc_id)
                        processed += 1

                        if result.success and result.act_text:
                            try:
                                # Check if document already exists in database
                                sejm_identifier = f"sejm_{doc_id}"
                                existing_doc = (
                                    pipeline.db_operations.get_document_by_eli_id(
                                        sejm_identifier
                                    )
                                )
                                if existing_doc:
                                    skipped += 1
                                else:
                                    # Store processed document to database using DocumentOperations.create_document
                                    doc_id_uuid = (
                                        pipeline.db_operations.create_document(
                                            title=result.metadata.get(
                                                "title", f"Sejm Document {doc_id}"
                                            ),
                                            content=result.act_text,
                                            document_type="sejm_document",
                                            eli_identifier=sejm_identifier,
                                            source_url=result.source_used or "",
                                            legal_act_type=result.metadata.get(
                                                "document_type", "sejm_proceeding"
                                            ),
                                            legal_domain="legislative",
                                        )
                                    )
                                    if doc_id_uuid:
                                        stored += 1
                                    else:
                                        failed += 1
                            except Exception:
                                failed += 1
                        else:
                            failed += 1

                        progress.update(task, advance=1)

                    except Exception as e:
                        failed += 1
                        self.logger.warning(f"Failed to process {doc_id}: {e}")
                        progress.update(task, advance=1)

                # Update progress as completed
                progress.update(
                    task, description="✅ Sejm ingestion completed", completed=True
                )

                self.console.print(
                    f"✅ [green]Sejm ingestion completed: {stored} documents stored[/green]"
                )

                return {
                    "processed": processed,
                    "stored": stored,
                    "skipped": skipped,
                    "failed": failed,
                    "documents_stored": stored,
                    "documents_failed": failed,
                    "documents_skipped": skipped,
                }

        except Exception as e:
            self.console.print(f"❌ [red]Sejm ingestion failed: {str(e)}[/red]")
            raise

    async def _execute_dual_stream_ingestion(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
        batch_size: int,
        force: bool,
    ) -> None:
        """Execute dual-stream document ingestion (ELI + Sejm streams separately)."""
        self.console.print(
            "🔄 [bold cyan]Starting dual-stream document ingestion (ELI stream + Sejm stream)...[/bold cyan]"
        )

        try:
            # Process ELI stream (enacted law documents)
            self.console.print(
                "📜 [bold green]Processing ELI stream (enacted law)...[/bold green]"
            )
            eli_result = await self._execute_eli_ingestion(
                start_date, end_date, limit // 2 if limit else None, batch_size, force
            )
            self._merge_stats(eli_result)

            # Process Sejm stream (legislative work documents)
            self.console.print(
                "⚖️ [bold blue]Processing Sejm stream (legislative work)...[/bold blue]"
            )
            sejm_result = await self._execute_sejm_ingestion(
                start_date, end_date, limit // 2 if limit else None, batch_size, force
            )
            self._merge_stats(sejm_result)

            # Add dual-stream breakdown stats
            self.stats["stream_breakdown"] = {
                "eli_stream": eli_result.get("processed", 0),
                "sejm_stream": sejm_result.get("processed", 0),
                "total_streams": 2,
            }

            self.console.print(
                "✅ [bold green]Dual-stream ingestion completed[/bold green]"
            )

            # Display stream results
            if hasattr(self, "stats") and "stream_breakdown" in self.stats:
                self.console.print()
                self.console.print(
                    "📊 [bold cyan]Dual-Stream Processing Results:[/bold cyan]"
                )
                stream_breakdown = self.stats["stream_breakdown"]
                self.console.print(
                    f"  📜 ELI stream (enacted law): {stream_breakdown.get('eli_stream', 0)} documents"
                )
                self.console.print(
                    f"  ⚖️ Sejm stream (legislative work): {stream_breakdown.get('sejm_stream', 0)} documents"
                )
                self.console.print(
                    f"  🔄 Total streams processed: {stream_breakdown.get('total_streams', 0)}"
                )

        except Exception as e:
            self.console.print(f"❌ [red]Dual-stream ingestion failed: {str(e)}[/red]")
            raise

    async def _get_eli_document_ids(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> List[str]:
        """Get ELI document IDs for processing (simplified for CLI)."""
        # For now, return sample document IDs for testing
        # In production, this would query the ELI API for actual document listings
        sample_ids = ["DU/2025/1", "DU/2025/2", "DU/2025/3", "MP/2025/1", "MP/2025/2"]

        if limit:
            return sample_ids[:limit]
        return sample_ids

    async def _get_sejm_document_ids(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: Optional[int],
    ) -> List[str]:
        """Get Sejm document IDs for processing (simplified for CLI)."""
        # For now, return sample document IDs for testing
        # In production, this would query the Sejm API for actual document listings
        sample_ids = ["10_1", "10_2", "10_3", "10_4", "10_5"]

        if limit:
            return sample_ids[:limit]
        return sample_ids

    def _merge_stats(self, pipeline_result: Dict[str, Any]) -> None:
        """Merge pipeline results into overall statistics."""
        self.stats["processed"] += pipeline_result.get("processed", 0)
        self.stats["stored"] += (
            pipeline_result.get("stored", 0)
            or pipeline_result.get("documents_stored", 0)
            or pipeline_result.get("documents_successful", 0)
        )
        self.stats["skipped"] += (
            pipeline_result.get("skipped", 0)
            or pipeline_result.get("documents_skipped", 0)
            or pipeline_result.get("documents_cached", 0)
        )
        self.stats["failed"] += pipeline_result.get("failed", 0) or pipeline_result.get(
            "documents_failed", 0
        )
