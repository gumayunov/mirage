#!/usr/bin/env bash
set -euo pipefail

# Show every command before execution
set -x

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
DOCUMENT_PATH="${1:-$PROJECT_ROOT/samples/the-art-of-war.epub}"
PROJECT_NAME="test-project"

MIRAGE="uv run --project $PROJECT_ROOT mirage"

# --- Create project ---
PROJECT_ID=$($MIRAGE projects create "$PROJECT_NAME")

# --- Add document ---
DOC_ID=$($MIRAGE documents add --project "$PROJECT_ID" "$DOCUMENT_PATH")

# --- Show document status ---
$MIRAGE documents status --project "$PROJECT_ID" "$DOC_ID"

# --- List all documents in the project ---
$MIRAGE documents list --project "$PROJECT_ID"
