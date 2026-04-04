#!/bin/bash
# 167a CLN vs Factor Path Ablation Study
# Ablation on epoch 20 checkpoint to determine if CLN competes with factor path
cd /home/max/Documents/vol-surface-vae-pub
PYTHONPATH=. python results/validations/2026-04-04/scripts/167a_cln_ablation.py
