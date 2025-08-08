#!/usr/bin/env python3
"""Generate document data flow diagram as PDF."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 16))
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis("off")

# Define colors
colors = {
    "entry": "#2196F3",  # Blue - Entry points
    "api": "#4CAF50",  # Green - API calls
    "process": "#FF9800",  # Orange - Processing
    "new": "#E91E63",  # Pink - Phase 6 new components
    "storage": "#9C27B0",  # Purple - Storage
    "cache": "#607D8B",  # Blue Grey - Cache
    "decision": "#FFC107",  # Amber - Decision points
}


def draw_box(x, y, width, height, text, color, text_size=8):
    """Draw a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="black",
        linewidth=1,
        alpha=0.8,
    )
    ax.add_patch(box)

    # Add text
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=text_size,
        weight="bold",
        wrap=True,
    )


def draw_diamond(x, y, width, height, text, color, text_size=8):
    """Draw a diamond shape for decisions."""
    diamond = patches.RegularPolygon(
        (x + width / 2, y + height / 2),
        4,
        radius=min(width, height) / 2,
        orientation=np.pi / 4,
        facecolor=color,
        edgecolor="black",
        linewidth=1,
        alpha=0.8,
    )
    ax.add_patch(diamond)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=text_size,
        weight="bold",
        wrap=True,
    )


def draw_arrow(x1, y1, x2, y2, label=""):
    """Draw an arrow between points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )


# Title
ax.text(
    10,
    15.5,
    "Document Data Flow: Sejm & ELI APIs → sejm_whiz Database Schema",
    ha="center",
    va="center",
    fontsize=16,
    weight="bold",
)

# Layer 1: Entry Points
draw_box(1, 14, 3, 0.8, "CLI Command\nsejm-whiz-cli ingest", colors["entry"])
draw_box(5, 14, 3, 0.8, "ingest.py\nDocument Processing", colors["entry"])
draw_box(9, 14, 3, 0.8, "CliPipelineOrchestrator\nExecution Control", colors["entry"])

# Layer 2: Source Selection
draw_diamond(8, 12.5, 3, 1, "Source\nType?", colors["decision"])
draw_box(1, 11.5, 2.5, 0.8, "multi-api", colors["process"])
draw_box(5, 11.5, 2.5, 0.8, "eli only", colors["process"])
draw_box(12, 11.5, 2.5, 0.8, "sejm only", colors["process"])

# Layer 3: Core Pipeline
draw_box(
    6,
    10,
    4,
    0.8,
    "DocumentIngestionPipeline\nBatch Processing + Caching",
    colors["process"],
)

# Layer 4: Document Processing
draw_box(3, 8.5, 4, 0.8, "ingest_documents_multi_source", colors["process"])
draw_box(8, 8.5, 4, 0.8, "process_document_multi_api", colors["process"])

# Layer 5: Cache Check
draw_diamond(7, 7, 2, 0.8, "Cache\nHit?", colors["cache"])
draw_box(11, 7, 2.5, 0.8, "Return\nCached Result", colors["cache"])

# Layer 6: Multi-API Processing
draw_box(
    5,
    5.5,
    4,
    0.8,
    "DualApiDocumentProcessor\nprocess_document_from_any_source",
    colors["api"],
)

# Layer 7: API Source Selection
draw_diamond(4, 4, 2, 0.8, "Source\nSelection", colors["decision"])
draw_box(1, 3, 2.5, 0.8, "Sejm API\n_try_sejm_api", colors["api"])
draw_box(7, 3, 2.5, 0.8, "ELI API\n_try_eli_api", colors["api"])
draw_box(13, 3, 2.5, 0.8, "Auto Select\nSejm → ELI", colors["api"])

# Layer 8: API Details
# Sejm API Flow
draw_box(0.5, 1.5, 1.8, 0.6, "get_act_with\n_full_text", colors["api"], 6)
draw_box(0.5, 0.8, 1.8, 0.6, "extract_act\n_metadata", colors["api"], 6)

# ELI API Flow
draw_box(6.5, 1.5, 1.8, 0.6, "HTML\nget_document\n_content", colors["api"], 6)
draw_box(8.5, 1.5, 1.8, 0.6, "PDF Fallback\nconvert_pdf\n_to_text", colors["api"], 6)

# NEW Phase 6 Components (Right side)
draw_box(
    16,
    6,
    3.5,
    0.8,
    "🆕 ContentExtractionOrchestrator\n4-Phase Guaranteed Processing",
    colors["new"],
)

# Phase 6 Alternative Sources
draw_box(15.5, 4.8, 2, 0.6, "Historical\nCache", colors["new"], 7)
draw_box(17.7, 4.8, 2, 0.6, "Wayback\nMachine", colors["new"], 7)
draw_box(15.5, 4, 2, 0.6, "Document\nRegistries", colors["new"], 7)
draw_box(17.7, 4, 2, 0.6, "Metadata\nReconstruction", colors["new"], 7)

# Quality Assessment
draw_box(
    15.5,
    2.8,
    4,
    0.8,
    "🆕 EnhancedContentValidator\nMulti-tier Quality Assessment",
    colors["new"],
)

# Manual Review
draw_box(
    15.5,
    1.8,
    4,
    0.8,
    "🆕 Manual Review Queue\nRich Context for Human Review",
    colors["new"],
)

# Storage Layer (Bottom)
draw_box(1, 0, 3, 0.6, "TextProcessor\nprocess_document", colors["storage"])
draw_box(5, 0, 3, 0.6, "DocumentOperations\nstore_legal_document", colors["storage"])
draw_box(9, 0, 3, 0.6, "PostgreSQL Database\nsejm_whiz_documents", colors["storage"])
draw_box(13, 0, 3, 0.6, "📄 sejm_whiz Schema\nNamespace Consistency", colors["storage"])

# Draw arrows for main flow
draw_arrow(2.5, 14, 5, 14.4)
draw_arrow(6.5, 14, 9, 14.4)
draw_arrow(10.5, 14, 9.5, 13.5)

# Source selection arrows
draw_arrow(8.5, 12.5, 2.5, 12)
draw_arrow(9, 12.5, 6.5, 12)
draw_arrow(9.5, 12.5, 13, 12)

# Pipeline flow
draw_arrow(8, 10.8, 8, 10)
draw_arrow(5, 9.3, 8, 9.3)
draw_arrow(10, 9.3, 10, 8.8)

# Cache decision
draw_arrow(8, 7.8, 8, 7.5)
draw_arrow(9, 7.4, 11, 7.4)

# Multi-API processing
draw_arrow(7, 6.5, 7, 6.3)

# Source selection
draw_arrow(5, 4.8, 5, 4.5)
draw_arrow(4.5, 4, 2.5, 3.8)
draw_arrow(5, 4, 7.5, 3.8)
draw_arrow(5.5, 4, 13.5, 3.8)

# API details
draw_arrow(2, 3, 1.5, 2.5)
draw_arrow(1.5, 1.5, 1.5, 1.4)
draw_arrow(8, 3, 7.5, 2.1)
draw_arrow(8.5, 3, 9.5, 2.1)

# Phase 6 enhancements
draw_arrow(12, 7, 16, 6.8)
draw_arrow(17.5, 6, 17.5, 5.4)
draw_arrow(17.5, 3.7, 17.5, 3.6)
draw_arrow(17.5, 2.8, 17.5, 2.6)

# Storage flow
draw_arrow(2.5, 0.6, 5, 0.3)
draw_arrow(6.5, 0.3, 9, 0.3)
draw_arrow(10.5, 0.3, 13, 0.3)

# Legend
legend_y = 13
ax.text(17, legend_y, "Legend:", fontsize=12, weight="bold")
legend_items = [
    ("Entry Points", colors["entry"]),
    ("API Calls", colors["api"]),
    ("Processing", colors["process"]),
    ("🆕 Phase 6 New", colors["new"]),
    ("Storage", colors["storage"]),
    ("Cache/Decisions", colors["cache"]),
]

for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y - 0.4 - (i * 0.3)
    draw_box(16.8, y_pos, 0.3, 0.2, "", color)
    ax.text(17.2, y_pos + 0.1, label, fontsize=8, va="center")

# Add phase annotations
ax.text(
    17.5,
    8,
    "Phase 6 Enhancements:",
    fontsize=10,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["new"], alpha=0.3),
)
ax.text(
    17.5,
    7.5,
    "• Alternative Sources\n• Quality Assessment\n• Manual Review\n• Guaranteed Outcome",
    fontsize=8,
    va="top",
)

# Database Schema Architecture
ax.text(
    0.5,
    13.5,
    "Database Schema:",
    fontsize=10,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["storage"], alpha=0.3),
)
schema_tables = [
    "sejm_whiz_documents",
    "sejm_whiz_amendments",
    "sejm_whiz_cross_references",
    "sejm_whiz_document_embeddings",
    "sejm_whiz_prediction_models",
]
schema_text = "\n".join([f"• {table}" for table in schema_tables])
ax.text(0.5, 12.8, schema_text, fontsize=7, va="top")

plt.tight_layout()
plt.savefig(
    "document_flow_diagram.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print(
    "✅ Document flow diagram updated with sejm_whiz schema and saved as document_flow_diagram.pdf"
)
