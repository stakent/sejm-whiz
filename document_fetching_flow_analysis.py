#!/usr/bin/env python3
"""Document Fetching Flow Analysis - Current vs Expected Behavior"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
fig.suptitle(
    "Document Fetching Flow: CURRENT (Broken) vs EXPECTED (Fixed)",
    fontsize=16,
    fontweight="bold",
)

# Colors
error_color = "#FF6B6B"
warning_color = "#FFD93D"
success_color = "#6BCF7F"
process_color = "#4ECDC4"
api_color = "#45B7D1"


def draw_box(ax, x, y, width, height, text, color, text_color="white"):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=text_color,
        wrap=True,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="black"):
    """Draw an arrow between two points"""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
    )


# CURRENT (BROKEN) FLOW - Left subplot
ax1.set_title(
    "🚨 CURRENT (BROKEN) FLOW", fontsize=14, fontweight="bold", color=error_color
)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis("off")

# CLI Command
draw_box(
    ax1,
    1,
    12,
    8,
    1.5,
    "CLI: sejm-whiz-cli.py ingest documents --since 100d",
    process_color,
)

# Pipeline Bridge
draw_box(
    ax1, 1, 10, 8, 1.5, "CliPipelineOrchestrator.execute_ingestion()", process_color
)
draw_arrow(ax1, 5, 12, 5, 11.5)

# Split to ELI and Sejm
draw_box(ax1, 0.5, 8, 3.5, 1.5, "_get_eli_document_ids()", warning_color)
draw_box(ax1, 5.5, 8, 3.5, 1.5, "_get_sejm_document_ids()", warning_color)
draw_arrow(ax1, 3, 10, 2.25, 9.5)
draw_arrow(ax1, 7, 10, 7.25, 9.5)

# The Problem - Hardcoded IDs
draw_box(
    ax1,
    0.5,
    6,
    3.5,
    1.5,
    "❌ HARDCODED:\n['DU/2025/1', 'DU/2025/2',\n'DU/2025/3', 'MP/2025/1',\n'MP/2025/2']",
    error_color,
)
draw_box(
    ax1,
    5.5,
    6,
    3.5,
    1.5,
    "❌ HARDCODED:\n['10_1', '10_2', '10_3',\n'10_4', '10_5']",
    error_color,
)
draw_arrow(ax1, 2.25, 8, 2.25, 7.5)
draw_arrow(ax1, 7.25, 8, 7.25, 7.5)

# Processing Results
draw_box(
    ax1,
    1,
    4,
    8,
    1.5,
    "Processing Result: Only 5 + 5 = 10 documents total\n(5 ELI skipped as duplicates, 5 Sejm failed)",
    error_color,
)
draw_arrow(ax1, 2.25, 6, 3, 5.5)
draw_arrow(ax1, 7.25, 6, 7, 5.5)

# API Reality (not accessed)
draw_box(
    ax1,
    1,
    2,
    8,
    1.5,
    "🌐 REALITY: ELI API has 1083 DU docs for 2025\nSejm API has hundreds of documents\n❌ NEVER QUERIED!",
    error_color,
    "white",
)
draw_arrow(ax1, 5, 4, 5, 3.5)

# Database Result
draw_box(ax1, 1, 0.2, 8, 1.2, "📊 DATABASE: Only 5 documents stored", error_color)
draw_arrow(ax1, 5, 2, 5, 1.4)

# EXPECTED (CORRECT) FLOW - Right subplot
ax2.set_title(
    "✅ EXPECTED (FIXED) FLOW", fontsize=14, fontweight="bold", color=success_color
)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis("off")

# CLI Command
draw_box(
    ax2,
    1,
    12,
    8,
    1.5,
    "CLI: sejm-whiz-cli.py ingest documents --since 100d",
    process_color,
)

# Pipeline Bridge
draw_box(
    ax2, 1, 10, 8, 1.5, "CliPipelineOrchestrator.execute_ingestion()", process_color
)
draw_arrow(ax2, 5, 12, 5, 11.5)

# Split to ELI and Sejm - FIXED
draw_box(
    ax2, 0.5, 8, 3.5, 1.5, "_get_eli_document_ids()\n✅ QUERIES ELI API", success_color
)
draw_box(
    ax2,
    5.5,
    8,
    3.5,
    1.5,
    "_get_sejm_document_ids()\n✅ QUERIES SEJM API",
    success_color,
)
draw_arrow(ax2, 3, 10, 2.25, 9.5)
draw_arrow(ax2, 7, 10, 7.25, 9.5)

# Actual API Calls
draw_box(
    ax2,
    0.5,
    6,
    3.5,
    1.5,
    "🌐 ELI API Search:\nGET /eli/acts/search\n?pubDateFrom=2025-04-30\n&pubDateTo=2025-08-08",
    api_color,
)
draw_box(
    ax2,
    5.5,
    6,
    3.5,
    1.5,
    "🌐 SEJM API Search:\nGET /sejm/term10/prints\nFilter by date range\n100+ documents",
    api_color,
)
draw_arrow(ax2, 2.25, 8, 2.25, 7.5)
draw_arrow(ax2, 7.25, 8, 7.25, 7.5)

# Processing Results
draw_box(
    ax2,
    1,
    4,
    8,
    1.5,
    "Processing Result: Hundreds of documents found\nDate-filtered, content extracted, quality validated",
    success_color,
)
draw_arrow(ax2, 2.25, 6, 3, 5.5)
draw_arrow(ax2, 7.25, 6, 7, 5.5)

# Real Data
draw_box(
    ax2,
    1,
    2,
    8,
    1.5,
    "📈 REAL RESULTS:\n• ELI: 50-200 documents (filtered by date)\n• Sejm: 20-100 documents (filtered by date)\n• Total: 100-300 documents expected",
    success_color,
)
draw_arrow(ax2, 5, 4, 5, 3.5)

# Database Result
draw_box(
    ax2, 1, 0.2, 8, 1.2, "📊 DATABASE: Hundreds of documents stored", success_color
)
draw_arrow(ax2, 5, 2, 5, 1.4)

# Add legend
problem_patch = mpatches.Patch(color=error_color, label="Problem/Error")
warning_patch = mpatches.Patch(color=warning_color, label="Issue/Warning")
process_patch = mpatches.Patch(color=process_color, label="Process Step")
api_patch = mpatches.Patch(color=api_color, label="API Call")
success_patch = mpatches.Patch(color=success_color, label="Fixed/Success")

fig.legend(
    handles=[problem_patch, warning_patch, process_patch, api_patch, success_patch],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=5,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(
    "/home/d/project/sejm-whiz/sejm-whiz-dev/document_fetching_flow_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/home/d/project/sejm-whiz/sejm-whiz-dev/document_fetching_flow_analysis.pdf",
    bbox_inches="tight",
)
print("📊 Document fetching flow analysis saved as PNG and PDF")

# Print the fix summary
print("\n🔧 FIXING THE ISSUE:")
print("=" * 50)
print(
    "PROBLEM: _get_eli_document_ids() and _get_sejm_document_ids() return hardcoded sample IDs"
)
print("SOLUTION: Replace with actual API calls to search for documents by date range")
print("\nRequired changes:")
print("1. ELI: Use EliApiClient.search_documents() with date filters")
print("2. Sejm: Use SejmApiClient to search for recent legislative documents")
print("3. Apply proper date filtering based on --since parameter")
print("4. Remove hardcoded sample document IDs")
print("\nExpected result: 100-300+ documents instead of 10")
