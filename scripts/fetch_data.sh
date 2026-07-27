#!/usr/bin/env bash
# Idempotent dataset setup. The container filesystem is ephemeral; git repo is not.
# Data lives in ./data (gitignored). Safe to re-run; skips anything already cached.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.bitgrad.data --all
