#!/usr/bin/env python3
"""
Progress tracking script for: Incremental API Refactoring
"""

import json
import sys
from datetime import datetime
from pathlib import Path


class RefactoringTracker:
    def __init__(self):
        self.progress_file = "refactoring_progress.json"
        self.plan_name = "Incremental API Refactoring"
        self.plan_strategy = "incremental"
        self.total_steps = 4
        self.step_ids = ["step_001", "step_002", "step_003", "step_004"]

    def load_progress(self):
        if Path(self.progress_file).exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {"started_at": None, "completed_steps": [], "current_step": None}

    def save_progress(self, progress):
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2, default=str)

    def start_tracking(self):
        progress = self.load_progress()
        if progress["started_at"] is None:
            progress["started_at"] = datetime.now().isoformat()
            self.save_progress(progress)
        print(f"Refactoring started at: {progress['started_at']}")

    def mark_step_complete(self, step_id):
        progress = self.load_progress()
        if step_id not in progress["completed_steps"]:
            progress["completed_steps"].append(step_id)
            progress["current_step"] = None
            self.save_progress(progress)
        print(f"Step {step_id} marked as complete")

    def mark_step_current(self, step_id):
        progress = self.load_progress()
        progress["current_step"] = step_id
        self.save_progress(progress)
        print(f"Step {step_id} marked as current")

    def show_status(self):
        progress = self.load_progress()
        completed = len(progress["completed_steps"])

        print(f"Refactoring Progress: {self.plan_name}")
        print(f"Strategy: {self.plan_strategy}")
        print(
            f"Progress: {completed}/{self.total_steps} steps completed ({completed/self.total_steps*100:.1f}%)"
        )

        if progress["started_at"]:
            started = datetime.fromisoformat(progress["started_at"])
            elapsed = datetime.now() - started
            print(f"Elapsed time: {elapsed}")

        if progress["current_step"]:
            print(f"Current step: {progress['current_step']}")

        print("\nCompleted steps:")
        for step_id in progress["completed_steps"]:
            print(f"  ✓ {step_id}")

        remaining_steps = [
            s for s in self.step_ids if s not in progress["completed_steps"]
        ]
        if remaining_steps:
            print("\nRemaining steps:")
            for step_id in remaining_steps:
                print(f"  ◦ {step_id}")


if __name__ == "__main__":
    tracker = RefactoringTracker()

    if len(sys.argv) < 2:
        tracker.show_status()
    elif sys.argv[1] == "start":
        tracker.start_tracking()
    elif sys.argv[1] == "complete" and len(sys.argv) > 2:
        tracker.mark_step_complete(sys.argv[2])
    elif sys.argv[1] == "current" and len(sys.argv) > 2:
        tracker.mark_step_current(sys.argv[2])
    else:
        print(
            "Usage: python track_progress.py [start|complete STEP_ID|current STEP_ID]"
        )
