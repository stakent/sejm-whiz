# 🚨 CRITICAL ISSUE: Document Fetching Returns Only 10 Documents Instead of 1000+

## Problem Summary

The CLI command `sejm-whiz-cli.py ingest documents --since 100d` is only processing 10 documents total, when the ELI API alone has **1,083 DU documents for 2025**.

## Root Cause: Hardcoded Sample Document IDs

The methods `_get_eli_document_ids()` and `_get_sejm_document_ids()` in `pipeline_bridge.py` return hardcoded sample IDs instead of querying the actual APIs.

## Current (Broken) Flow

```
CLI Command: --since 100d
         ↓
CliPipelineOrchestrator.execute_ingestion()
         ↓
    ┌─────────────────────┬─────────────────────┐
    ↓                     ↓
_get_eli_document_ids()  _get_sejm_document_ids()
    ↓                     ↓
❌ HARDCODED:            ❌ HARDCODED:
["DU/2025/1",           ["10_1", "10_2",
 "DU/2025/2",            "10_3", "10_4",
 "DU/2025/3",            "10_5"]
 "MP/2025/1",
 "MP/2025/2"]
    ↓                     ↓
Processing: 5 docs       Processing: 5 docs
5 skipped (duplicates)   5 failed
    ↓                     ↓
        Result: Only 10 documents total
        📊 Database: Only 5 documents stored

🌐 REALITY: ELI API has 1083+ documents - NEVER QUERIED!
```

## Expected (Fixed) Flow

```
CLI Command: --since 100d
         ↓
CliPipelineOrchestrator.execute_ingestion()
         ↓
    ┌─────────────────────┬─────────────────────┐
    ↓                     ↓
_get_eli_document_ids()  _get_sejm_document_ids()
    ↓                     ↓
✅ ELI API SEARCH:       ✅ SEJM API SEARCH:
GET /eli/acts/search     GET /sejm/term10/prints
?pubDateFrom=2025-04-30  Filter by date range
&pubDateTo=2025-08-08
&limit=500
    ↓                     ↓
Returns: 50-200 docs     Returns: 20-100 docs
(filtered by date)       (filtered by date)
    ↓                     ↓
Processing: Real docs    Processing: Real docs
Content extraction       Content extraction
Quality validation       Quality validation
    ↓                     ↓
        Result: 100-300+ documents processed
        📊 Database: Hundreds of documents stored
```

## Code Fix Required

### File: `components/sejm_whiz/cli/commands/pipeline_bridge.py`

**Lines 412-418 (ELI method):**

```python
# CURRENT - BROKEN:
async def _get_eli_document_ids(self, start_date, end_date, limit) -> List[str]:
    """Get ELI document IDs for processing (simplified for CLI)."""
    # For now, return sample document IDs for testing
    sample_ids = ["DU/2025/1", "DU/2025/2", "DU/2025/3", "MP/2025/1", "MP/2025/2"]
    if limit:
        return sample_ids[:limit]
    return sample_ids
```

**NEEDS TO BE:**

```python
# FIXED:
async def _get_eli_document_ids(self, start_date, end_date, limit) -> List[str]:
    """Get ELI document IDs by searching the API with date filters."""
    from sejm_whiz.eli_api.client import EliApiClient

    async with EliApiClient() as client:
        search_result = await client.search_documents(
            date_from=start_date,
            date_to=end_date,
            limit=limit or 500
        )

        # Extract ELI IDs from search results
        document_ids = []
        for doc in search_result.documents:
            if doc.eli_id:
                document_ids.append(doc.eli_id)

        return document_ids
```

**Lines 426-433 (Sejm method):**

```python
# CURRENT - BROKEN:
async def _get_sejm_document_ids(self, start_date, end_date, limit) -> List[str]:
    """Get Sejm document IDs for processing (simplified for CLI)."""
    # For now, return sample document IDs for testing
    sample_ids = ["10_1", "10_2", "10_3", "10_4", "10_5"]
    if limit:
        return sample_ids[:limit]
    return sample_ids
```

**NEEDS TO BE:**

```python
# FIXED:
async def _get_sejm_document_ids(self, start_date, end_date, limit) -> List[str]:
    """Get Sejm document IDs by searching legislative documents."""
    from sejm_whiz.sejm_api.client import SejmApiClient

    async with SejmApiClient() as client:
        # Get current term (10) and search for prints/documents
        current_term = await client.get_current_term()

        # This would need to be implemented to search Sejm documents by date
        # For now, get a reasonable number of recent document IDs
        # Implementation depends on available Sejm API endpoints for date filtering

        # Placeholder - needs actual Sejm API integration for document search
        document_ids = []
        # TODO: Implement actual Sejm document search with date filtering

        return document_ids
```

## Impact Analysis

### Current State

- ❌ **Only 10 documents processed** (5 ELI + 5 Sejm)
- ❌ **Missing 1000+ available documents**
- ❌ **Date filtering not working** (ignores --since parameter)
- ❌ **Poor user experience** (appears broken)

### After Fix

- ✅ **100-300+ documents processed** per ingestion run
- ✅ **Real date filtering** based on --since parameter
- ✅ **Full API utilization** of both ELI and Sejm sources
- ✅ **Production-ready** document ingestion

## Urgency: CRITICAL

This issue makes the document ingestion system completely non-functional for production use. The system appears to work but is only processing a tiny fraction of available documents.

## Next Steps

1. **Fix `_get_eli_document_ids()`** - Replace hardcoded IDs with ELI API search
1. **Fix `_get_sejm_document_ids()`** - Replace hardcoded IDs with Sejm API search
1. **Test with real date ranges** - Verify hundreds of documents are found
1. **Verify database storage** - Confirm all found documents get processed and stored

## Testing Commands

After fix, these should return many more documents:

```bash
# Should find 50-200 ELI documents from last 100 days
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --source eli --since 100d

# Should find 20-100 Sejm documents from last 100 days
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --source sejm --since 100d

# Should find 100-300+ total documents
DEPLOYMENT_ENV=p7 uv run python sejm-whiz-cli.py ingest documents --since 100d
```
