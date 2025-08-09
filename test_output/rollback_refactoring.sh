#!/bin/bash
# Rollback script for: Incremental API Refactoring

echo "ROLLBACK SCRIPT FOR: Incremental API Refactoring"
echo "WARNING: This will undo refactoring changes"
echo ""

# Show available checkpoints
echo "Available checkpoints:"
git tag -l "checkpoint-*" | sort
echo ""

echo "Recent commits:"
git log --oneline -10
echo ""

read -p "Choose rollback method (1=latest checkpoint, 2=specific commit, 3=full reset): " choice

case $choice in
    1)
        latest_checkpoint=$(git tag -l "checkpoint-*" | sort | tail -1)
        if [ -z "$latest_checkpoint" ]; then
            echo "No checkpoints found"
            exit 1
        fi
        echo "Rolling back to: $latest_checkpoint"
        git reset --hard "$latest_checkpoint"
        ;;
    2)
        read -p "Enter commit hash: " commit_hash
        echo "Rolling back to: $commit_hash"
        git reset --hard "$commit_hash"
        ;;
    3)
        echo "Full reset to backup tag"
        git reset --hard backup-before-refactor
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Clean up
git clean -fd

echo "Rollback completed"
echo "Run validation to ensure system is functional:"
echo "./validate_refactoring.sh"
