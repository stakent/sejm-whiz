"""Search operations commands."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
from uuid import UUID

from sejm_whiz.semantic_search.core import (
    process_search_query,
    search_similar_documents,
    get_semantic_search_service,
)
from sejm_whiz.semantic_search.ranker import RankingStrategy
from sejm_whiz.logging import get_enhanced_logger

console = Console()
app = typer.Typer(no_args_is_help=False)
logger = get_enhanced_logger(__name__)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """🔍 Search operations."""
    if ctx.invoked_subcommand is None:
        # Run status command by default
        status()


@app.command()
def query(
    text: str = typer.Argument(..., help="Search query text"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    min_score: float = typer.Option(
        0.7, "--min-score", "-s", help="Minimum similarity score"
    ),
    document_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Document type filter"
    ),
):
    """🔍 Search for documents using semantic similarity."""
    try:
        console.print(f"🔍 Searching for: [bold cyan]'{text}'[/bold cyan]")

        # Perform semantic search
        results = process_search_query(
            query=text,
            limit=limit,
            document_type=document_type,
            ranking_strategy=RankingStrategy.COMPOSITE,
        )

        # Filter by minimum score
        filtered_results = [
            r for r in results if r.get("similarity_score", 0) >= min_score
        ]

        if not filtered_results:
            console.print(
                "❌ [yellow]No documents found matching your criteria.[/yellow]"
            )
            console.print(f"  • Try lowering --min-score (current: {min_score})")
            console.print(
                "  • Remove document type filter"
                + (f" (current: {document_type})" if document_type else "")
            )
            return

        # Display results
        table = Table(title=f"Search Results ({len(filtered_results)} documents)")
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="blue", width=12)
        table.add_column("Title", style="white", min_width=30)
        table.add_column("Published", style="dim", width=12)

        for result in filtered_results:
            score_text = f"{result['similarity_score']:.3f}"
            doc_type = result.get("document_type", "Unknown")[:10]
            title = result.get("title", "No title")[:60] + (
                "..." if len(result.get("title", "")) > 60 else ""
            )
            published = (
                result.get("published_at", "")[:10]
                if result.get("published_at")
                else "Unknown"
            )

            table.add_row(score_text, doc_type, title, published)

        console.print(table)

        # Show passages for top result
        if filtered_results and filtered_results[0].get("matched_passages"):
            console.print("\n📄 [bold]Top Result Passages:[/bold]")
            for i, passage in enumerate(filtered_results[0]["matched_passages"][:2], 1):
                console.print(
                    f"  {i}. {passage[:200]}..."
                    if len(passage) > 200
                    else f"  {i}. {passage}"
                )

    except Exception as e:
        logger.error(f"Search query failed: {e}")
        console.print(f"❌ [bold red]Search failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def similar(
    document_id: str = typer.Argument(
        ..., help="Document ID to find similar documents for"
    ),
    limit: int = typer.Option(
        5, "--limit", "-l", help="Maximum similar documents to return"
    ),
    threshold: float = typer.Option(
        0.6, "--threshold", "-t", help="Similarity threshold"
    ),
):
    """🔗 Find documents similar to a given document."""
    try:
        # Validate UUID format
        try:
            UUID(document_id)
        except ValueError:
            console.print(
                f"❌ [bold red]Invalid document ID format: {document_id}[/bold red]"
            )
            console.print("  • Document ID must be a valid UUID")
            raise typer.Exit(1)

        console.print(
            f"🔗 Finding documents similar to: [bold cyan]{document_id}[/bold cyan]"
        )

        # Find similar documents
        results = search_similar_documents(
            document_id=document_id,
            limit=limit,
            ranking_strategy=RankingStrategy.TEMPORAL_BOOST,
        )

        # Filter by threshold
        filtered_results = [
            r for r in results if r.get("similarity_score", 0) >= threshold
        ]

        if not filtered_results:
            console.print("❌ [yellow]No similar documents found.[/yellow]")
            console.print(f"  • Try lowering --threshold (current: {threshold})")
            console.print("  • Ensure document ID exists in database")
            return

        # Display results
        table = Table(title=f"Similar Documents ({len(filtered_results)} found)")
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="blue", width=12)
        table.add_column("Title", style="white", min_width=30)
        table.add_column("Published", style="dim", width=12)
        table.add_column("Document ID", style="dim", width=36)

        for result in filtered_results:
            score_text = f"{result['similarity_score']:.3f}"
            doc_type = result.get("document_type", "Unknown")[:10]
            title = result.get("title", "No title")[:50] + (
                "..." if len(result.get("title", "")) > 50 else ""
            )
            published = (
                result.get("published_at", "")[:10]
                if result.get("published_at")
                else "Unknown"
            )
            doc_id = result.get("document_id", "")[:8] + "..."

            table.add_row(score_text, doc_type, title, published, doc_id)

        console.print(table)

    except Exception as e:
        logger.error(f"Similar documents search failed: {e}")
        console.print(f"❌ [bold red]Similar documents search failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def reindex(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force complete reindexing"
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", "-b", help="Indexing batch size"
    ),
):
    """🔄 Rebuild the search index."""
    console.print(
        "❌ [bold red]Search index rebuilding is not implemented yet.[/bold red]"
    )
    console.print("This command would rebuild the search index but requires:")
    console.print("  • Document embeddings generation pipeline")
    console.print("  • Vector database integration")
    console.print("  • Batch processing and progress tracking")
    console.print("  • Index optimization and validation")
    console.print(f"  • Force mode: {force}")
    console.print(f"  • Batch size: {batch_size}")
    raise typer.Exit(1)


@app.command()
def benchmark(
    queries: int = typer.Option(100, "--queries", "-q", help="Number of test queries"),
    concurrent: int = typer.Option(
        10, "--concurrent", "-c", help="Concurrent requests"
    ),
):
    """📊 Run search performance benchmarks."""
    console.print("❌ [bold red]Search benchmarking is not implemented yet.[/bold red]")
    console.print("This command would run search performance tests but requires:")
    console.print("  • Search system implementation")
    console.print("  • Performance testing framework")
    console.print("  • Query load generation")
    console.print("  • Metrics collection and analysis")
    console.print(f"  • Test queries: {queries}")
    console.print(f"  • Concurrent requests: {concurrent}")
    raise typer.Exit(1)


@app.command()
def status():
    """📊 Show search system status."""
    try:
        console.print("📊 [bold]Search System Status[/bold]")
        console.print()

        # Get comprehensive statistics
        service = get_semantic_search_service()
        stats = service.get_search_statistics()

        if not stats:
            console.print(
                "❌ [yellow]Could not retrieve search system statistics[/yellow]"
            )
            return

        # Indexing statistics
        if "indexing" in stats:
            indexing_stats = stats["indexing"]
            console.print("🗂️  [bold blue]Document Index[/bold blue]")
            console.print(
                f"  • Total documents: {indexing_stats.get('total_documents', 'Unknown')}"
            )
            console.print(
                f"  • Documents with embeddings: {indexing_stats.get('embedded_documents', 'Unknown')}"
            )
            console.print(
                f"  • Index coverage: {indexing_stats.get('coverage_percentage', 'Unknown')}%"
            )
            console.print()

        # Search engine status
        if "search_engine" in stats:
            search_stats = stats["search_engine"]
            console.print("🔍 [bold blue]Search Engine[/bold blue]")
            console.print(
                f"  • Status: {'✅ Available' if search_stats.get('available') else '❌ Unavailable'}"
            )
            console.print(
                f"  • Embedding model: {search_stats.get('embedding_model', 'Unknown')}"
            )
            console.print(
                f"  • Vector dimensions: {search_stats.get('embedding_dimensions', 'Unknown')}"
            )
            console.print()

        # Ranking system status
        if "ranking" in stats:
            ranking_stats = stats["ranking"]
            console.print("🏆 [bold blue]Ranking System[/bold blue]")
            console.print(f"  • Strategy: {ranking_stats.get('strategy', 'Unknown')}")
            console.print(
                f"  • Temporal decay: {ranking_stats.get('temporal_decay_days', 'Unknown')} days"
            )
            if ranking_stats.get("document_type_weights"):
                console.print("  • Document type weights configured")
            console.print()

        # Cross-register analysis
        if "cross_register" in stats:
            cross_stats = stats["cross_register"]
            console.print("🔗 [bold blue]Cross-Register Analysis[/bold blue]")
            console.print(
                f"  • Status: {'✅ Available' if cross_stats.get('available') else '❌ Unavailable'}"
            )
            console.print(
                f"  • Register mappings: {cross_stats.get('mapping_count', 'Unknown')}"
            )
            console.print()

        # System health summary
        total_docs = stats.get("indexing", {}).get("total_documents", 0)
        embedded_docs = stats.get("indexing", {}).get("embedded_documents", 0)

        if total_docs > 0 and embedded_docs > 0:
            console.print("✅ [bold green]Search system is operational[/bold green]")
        elif total_docs > 0:
            console.print(
                "⚠️  [bold yellow]Search system partially operational (missing embeddings)[/bold yellow]"
            )
        else:
            console.print(
                "❌ [bold red]Search system not operational (no documents)[/bold red]"
            )

    except Exception as e:
        logger.error(f"Failed to get search status: {e}")
        console.print(f"❌ [bold red]Failed to get search status: {e}[/bold red]")
        raise typer.Exit(1)
