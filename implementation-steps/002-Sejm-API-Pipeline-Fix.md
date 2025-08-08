# IMPLEMENTATION BRIEF #002

Priority: High
Category: Backend

## Requirement:

- Fix Sejm API integration in CLI pipeline bridge to enable both ELI and Sejm data sources

## Acceptance Criteria:

- [ ] `enhanced_data_processor.py` runs without import errors
- [ ] CLI bridge shows Sejm as "implemented" instead of "not yet implemented"
- [ ] `uv run python sejm-whiz-cli.py ingest documents --source sejm` works
- [ ] Sejm ingestion processes parliamentary proceedings and stores in database
- [ ] Integration maintains existing error handling and progress reporting
- [ ] No regression in existing ELI API functionality

## Technical Constraints:

- Timeline: 1-2 hours
- Dependencies: Fix missing `sejm_whiz.data_pipeline` module reference
- Performance requirements: Maintain existing 4+ docs/sec throughput
- Compatibility requirements: Must work with existing Redis and PostgreSQL setup

## Scope Boundaries:

- Do NOT modify Sejm API client implementation
- Do NOT change database models or operations
- Do NOT alter existing CLI parameter structure
- Focus only on fixing import errors and pipeline bridge integration
