#!/bin/bash
# 140a Attention Pattern & Noise Pathway Analysis
# Analyzes transformer attention patterns, noise pathway, and cross-cell correlation mechanism
set -euo pipefail

cd /home/max/Documents/vol-surface-vae-pub

echo "=== 140a Attention Analysis ==="
echo "Started: $(date)"

PYTHONPATH=. python results/validations/2026-03-22/scripts/140a_attention_analysis.py

echo "Completed: $(date)"
