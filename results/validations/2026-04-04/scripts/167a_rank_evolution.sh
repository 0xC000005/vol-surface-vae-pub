#!/bin/bash
# 167a Noise Effective Rank Evolution Analysis
# Tracks eff_rank, L_norm, pathway spread across ep10-ep80 checkpoints
set -euo pipefail
cd /home/max/Documents/vol-surface-vae-pub
PYTHONPATH=. python results/validations/2026-04-04/scripts/167a_rank_evolution.py
