#!/bin/bash
# 167a Gradient Analysis: Verify claim that CRPS gradient starvation kills load_head
# Compares gradient magnitudes to base_head vs load_head at epoch 20 and epoch 60
# Also checks if weight decay removal > gradient addition

set -euo pipefail
cd /home/max/Documents/vol-surface-vae-pub

PYTHONPATH=. python -u results/validations/2026-04-04/scripts/167a_gradient_analysis.py
