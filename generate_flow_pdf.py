#!/usr/bin/env python3
"""Generate document data flow diagram as PDF using reportlab."""

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


def create_flow_diagram():
    # Create PDF document in landscape orientation
    doc = SimpleDocTemplate("document_flow_diagram.pdf", pagesize=landscape(A4))
    story = []

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=30,
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue,
    )

    # Title
    story.append(
        Paragraph(
            "Document Data Flow: Sejm & ELI APIs → Final Text Storage", title_style
        )
    )
    story.append(Spacer(1, 20))

    # Flow description
    flow_data = [
        ["Layer", "Component", "Description", "Type"],
        ["1. Entry", "CLI Command", "sejm-whiz-cli ingest documents", "Entry Point"],
        ["", "ingest.py", "Document processing commands", "Entry Point"],
        [
            "",
            "CliPipelineOrchestrator",
            "Execution control and progress",
            "Entry Point",
        ],
        ["2. Routing", "Source Decision", "multi-api / eli / sejm / both", "Decision"],
        [
            "3. Pipeline",
            "CachedDocumentIngestionPipeline",
            "Batch processing with caching",
            "Core",
        ],
        [
            "",
            "ingest_documents_multi_source",
            "Multi-source document ingestion",
            "Core",
        ],
        ["", "process_document_multi_api", "Individual document processing", "Core"],
        ["4. Cache", "Cache Check", "Redis-based response caching", "Cache"],
        [
            "5. Multi-API",
            "MultiApiDocumentProcessor",
            "Unified API processing logic",
            "API Layer",
        ],
        [
            "",
            "process_document_from_any_source",
            "Source selection and fallback",
            "API Layer",
        ],
        [
            "6. Sejm Path",
            "SejmApiClient.get_act_with_full_text",
            "Fetch act text from Sejm API",
            "Sejm API",
        ],
        ["", "extract_act_metadata", "Extract document metadata", "Sejm API"],
        ["", "is_sejm_content_complete", "Validate content completeness", "Sejm API"],
        [
            "7. ELI Path",
            "EliApiClient.get_document_content_with_basic_fallback",
            "HTML→PDF fallback",
            "ELI API",
        ],
        ["", "get_document_content(html)", "Try HTML content first", "ELI API"],
        ["", "get_document_content(pdf)", "PDF fallback conversion", "ELI API"],
        ["", "content_validator validation", "Quality checks for usability", "ELI API"],
        [
            "8. 🆕 Phase 6",
            "ContentExtractionOrchestrator",
            "4-phase guaranteed processing",
            "NEW",
        ],
        [
            "",
            "AlternativeContentSources",
            "Wayback, registries, cache, metadata",
            "NEW",
        ],
        ["", "try_cached_historical_content", "Check internal cache history", "NEW"],
        ["", "try_wayback_machine", "Archive.org historical snapshots", "NEW"],
        ["", "try_document_registry_apis", "ISAP, Lex legal databases", "NEW"],
        [
            "",
            "extract_from_document_references",
            "Metadata-based reconstruction",
            "NEW",
        ],
        [
            "9. 🆕 Quality",
            "EnhancedContentValidator",
            "Multi-tier quality assessment",
            "NEW",
        ],
        ["", "assess_content_quality", "High/Medium/Low/Summary tiers", "NEW"],
        [
            "",
            "Polish legal pattern detection",
            "Language and structure validation",
            "NEW",
        ],
        [
            "10. 🆕 Manual",
            "Manual Review Queue",
            "Rich context for human review",
            "NEW",
        ],
        [
            "",
            "_prepare_manual_review_context",
            "Failure analysis and suggestions",
            "NEW",
        ],
        [
            "11. Processing",
            "TextProcessor.process_document",
            "Text cleaning and structuring",
            "Processing",
        ],
        [
            "12. Storage",
            "DocumentOperations.store_legal_document",
            "Database persistence",
            "Storage",
        ],
        [
            "",
            "PostgreSQL LegalDocument table",
            "Final document text storage",
            "Storage",
        ],
    ]

    # Create table
    table = Table(flow_data, colWidths=[0.8 * inch, 2.2 * inch, 3 * inch, 1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                # Color coding by type
                ("BACKGROUND", (0, 1), (-1, 4), colors.lightblue),  # Entry points
                ("BACKGROUND", (0, 11), (-1, 17), colors.lightgreen),  # APIs
                ("BACKGROUND", (0, 18), (-1, 28), colors.pink),  # NEW Phase 6
                ("BACKGROUND", (0, 29), (-1, -1), colors.plum),  # Storage
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 20))

    # Key Features section
    story.append(Paragraph("🔑 Key Data Flow Features", heading_style))

    features_data = [
        ["Feature", "Description", "Benefit"],
        [
            "Multi-API Processing",
            "Sejm API → ELI API → Alternative Sources",
            "Maximum content coverage",
        ],
        [
            "Guaranteed Outcome",
            "Every document gets processed OR flagged for manual review",
            "Zero document loss",
        ],
        [
            "Quality Tiers",
            "High/Medium/Low/Summary quality assessment",
            "Content quality transparency",
        ],
        [
            "Alternative Sources",
            "Wayback Machine, document registries, metadata reconstruction",
            "404 error recovery",
        ],
        [
            "Comprehensive Caching",
            "API responses, processed content, multi-level cache",
            "Performance optimization",
        ],
        [
            "Rich Manual Context",
            "Failure analysis, suggestions, priority assessment",
            "Efficient human review",
        ],
        [
            "Polish Legal Patterns",
            "Language detection, legal structure validation",
            "Domain-specific quality",
        ],
    ]

    features_table = Table(features_data, colWidths=[1.5 * inch, 3.5 * inch, 2 * inch])
    features_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightcyan),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(features_table)
    story.append(Spacer(1, 20))

    # Processing Guarantee section
    story.append(
        Paragraph("🎯 Processing Guarantee (Phase 6 Enhancement)", heading_style)
    )

    guarantee_text = """
    <b>Before Phase 6:</b> Documents returning 404 errors would disappear without processing<br/>
    <b>After Phase 6:</b> Every document gets one of these outcomes:<br/>
    <br/>
    • ✅ <b>Success:</b> Text extracted and stored in database<br/>
    • 🔄 <b>Alternative Success:</b> Content found via Wayback Machine or other sources<br/>
    • 📄 <b>Summary Success:</b> Metadata-based summary generated<br/>
    • 👥 <b>Manual Review:</b> Flagged with rich context for human processing<br/>
    <br/>
    <b>Result:</b> 0% unprocessed documents - complete processing guarantee
    """

    story.append(Paragraph(guarantee_text, styles["Normal"]))

    # Build PDF
    doc.build(story)
    print("✅ Document flow diagram saved as document_flow_diagram.pdf")


if __name__ == "__main__":
    create_flow_diagram()
