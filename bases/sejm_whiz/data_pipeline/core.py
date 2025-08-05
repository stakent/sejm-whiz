"""Data pipeline base for batch processing workflows."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Generic
import logging
from datetime import datetime

# Type variables for pipeline data flow
InputData = TypeVar("InputData")
OutputData = TypeVar("OutputData")


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""

    pass


class PipelineStep(ABC, Generic[InputData, OutputData]):
    """Abstract base class for pipeline steps with generic types."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")

    @abstractmethod
    async def process(self, data: InputData) -> OutputData:
        """Process data in this pipeline step."""
        pass

    async def validate_input(self, data: InputData) -> bool:
        """Validate input data before processing."""
        return True

    async def validate_output(self, data: OutputData) -> bool:
        """Validate output data after processing."""
        return True


class DataPipeline:
    """Main data pipeline orchestrator for batch processing."""

    def __init__(self, name: str, steps: List[PipelineStep]):
        self.name = name
        self.steps = steps
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.runs = 0
        self.successes = 0
        self.failures = 0
        self.last_run: datetime | None = None
        self.last_success: datetime | None = None

    async def run(self, initial_data: Any) -> Any:
        """Run the complete pipeline."""
        self.runs += 1
        start_time = datetime.now()
        self.last_run = start_time

        try:
            self.logger.info(
                f"Starting pipeline '{self.name}' with {len(self.steps)} steps"
            )

            current_data = initial_data

            for i, step in enumerate(self.steps):
                self.logger.info(
                    f"Executing step {i + 1}/{len(self.steps)}: {step.name}"
                )

                # Validate input
                if not await step.validate_input(current_data):
                    raise DataPipelineError(
                        f"Input validation failed for step '{step.name}'"
                    )

                # Process data
                current_data = await step.process(current_data)

                # Validate output
                if not await step.validate_output(current_data):
                    raise DataPipelineError(
                        f"Output validation failed for step '{step.name}'"
                    )

                self.logger.info(f"Step '{step.name}' completed successfully")

            self.successes += 1
            self.last_success = datetime.now()
            duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Pipeline '{self.name}' completed successfully in {duration:.2f}s"
            )
            return current_data

        except Exception as e:
            self.failures += 1
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Pipeline '{self.name}' failed after {duration:.2f}s: {str(e)}"
            )
            raise DataPipelineError(f"Pipeline '{self.name}' failed: {str(e)}") from e

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        return {
            "runs": self.runs,
            "successes": self.successes,
            "failures": self.failures,
            "last_run": self.last_run,
            "last_success": self.last_success,
        }


class BatchProcessor:
    """Batch processor for handling multiple data items."""

    def __init__(self, pipeline: DataPipeline, batch_size: int = 100):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.logger = logging.getLogger(f"batch_processor.{pipeline.name}")

    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Process a batch of items through the pipeline."""
        results = []
        failed_items = []

        self.logger.info(f"Processing batch of {len(items)} items")

        for i, item in enumerate(items):
            try:
                result = await self.pipeline.run(item)
                results.append(result)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(items)} items")

            except Exception as e:
                self.logger.error(f"Failed to process item {i}: {str(e)}")
                failed_items.append((i, item, str(e)))

        if failed_items:
            self.logger.warning(
                f"Failed to process {len(failed_items)} items out of {len(items)}"
            )

        self.logger.info(
            f"Batch processing completed: {len(results)} successes, {len(failed_items)} failures"
        )
        return results

    async def process_stream(self, items: List[Any]) -> List[Any]:
        """Process items in batches of specified size."""
        all_results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)

        return all_results
