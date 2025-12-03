#!/bin/bash
#
# Generate VAE AR Multi-Step Sequences - All Periods for Context60 Model
#
# Generates autoregressive sequences for all periods:
# - crisis: ~30 min (2 horizons: 180, 270 days)
# - insample: ~120 min (2 horizons)
# - oos: ~30 min (2 horizons)
# - gap: ~35 min (2 horizons)
# Total: ~150-180 minutes per sampling mode
#
# Supports two sampling modes:
# - oracle: Posterior sampling (approximate oracle for AR)
# - prior: Realistic sampling without future knowledge
#
# Usage:
#   # Prior sampling (default, realistic)
#   bash experiments/backfill/context60/autoregressive/run_generate_all_ar_sequences.sh prior
#
#   # Oracle sampling
#   bash experiments/backfill/context60/autoregressive/run_generate_all_ar_sequences.sh oracle
#
#   # Run in background (recommended)
#   nohup bash experiments/backfill/context60/autoregressive/run_generate_all_ar_sequences.sh prior > vae_ar_context60_prior.log 2>&1 &
#

set -e  # Exit on error

# Get sampling mode from first argument (default: prior)
SAMPLING_MODE="${1:-prior}"

if [[ "$SAMPLING_MODE" != "oracle" && "$SAMPLING_MODE" != "prior" ]]; then
    echo "ERROR: Invalid sampling mode '$SAMPLING_MODE'"
    echo "Usage: $0 [oracle|prior]"
    exit 1
fi

# Colors for output (if running in terminal)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOGFILE="vae_ar_context60_generation_${SAMPLING_MODE}_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log_section() {
    echo "" | tee -a "$LOGFILE"
    echo "================================================================================" | tee -a "$LOGFILE"
    echo "$1" | tee -a "$LOGFILE"
    echo "================================================================================" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
}

# Start
log_section "VAE AUTOREGRESSIVE MULTI-STEP GENERATION (CONTEXT60) - START"
log "Sampling mode: ${SAMPLING_MODE}"
log "Logfile: $LOGFILE"
log "Script: $0"
log "Working directory: $(pwd)"
echo ""

# Check if we're in the right directory
if [ ! -f "experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py" ]; then
    log "${RED}ERROR: Must run from repository root!${NC}"
    log "Current directory: $(pwd)"
    log "Expected file: experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py"
    exit 1
fi

# Check if model exists
if [ ! -f "models/backfill/context60_experiment/checkpoints/backfill_context60_best.pt" ]; then
    log "${RED}ERROR: Model file not found!${NC}"
    log "Expected: models/backfill/context60_experiment/checkpoints/backfill_context60_best.pt"
    log "Note: You may need to use a phase checkpoint like backfill_context60_phase4_ep599.pt"
    exit 1
fi

# Check if data exists
if [ ! -f "data/vol_surface_with_ret.npz" ]; then
    log "${RED}ERROR: Data file not found!${NC}"
    log "Expected: data/vol_surface_with_ret.npz"
    exit 1
fi

log "${GREEN}✓ All required files found${NC}"
echo ""

# Start time
START_TIME=$(date +%s)

# ============================================================================
# Generate Crisis Period
# ============================================================================

log_section "GENERATING CRISIS PERIOD (2008-2010) - AR"
log "Sampling mode: ${SAMPLING_MODE}"
log "Expected runtime: ~30 minutes"
log "Output: 2 files (180-day, 270-day AR sequences)"
echo ""

CRISIS_START=$(date +%s)
PYTHONPATH=. python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py \
    --period crisis \
    --sampling_mode ${SAMPLING_MODE} \
    2>&1 | tee -a "$LOGFILE"
CRISIS_STATUS=$?
CRISIS_END=$(date +%s)
CRISIS_DURATION=$((CRISIS_END - CRISIS_START))

if [ $CRISIS_STATUS -eq 0 ]; then
    log "${GREEN}✓ Crisis period complete${NC} (${CRISIS_DURATION}s)"
else
    log "${RED}✗ Crisis period FAILED${NC} (exit code: $CRISIS_STATUS)"
    exit $CRISIS_STATUS
fi
echo ""

# ============================================================================
# Generate Insample Period
# ============================================================================

log_section "GENERATING INSAMPLE PERIOD (2004-2019) - AR"
log "Sampling mode: ${SAMPLING_MODE}"
log "Expected runtime: ~120 minutes"
log "Output: 2 files (180-day, 270-day AR sequences)"
echo ""

INSAMPLE_START=$(date +%s)
PYTHONPATH=. python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py \
    --period insample \
    --sampling_mode ${SAMPLING_MODE} \
    2>&1 | tee -a "$LOGFILE"
INSAMPLE_STATUS=$?
INSAMPLE_END=$(date +%s)
INSAMPLE_DURATION=$((INSAMPLE_END - INSAMPLE_START))

if [ $INSAMPLE_STATUS -eq 0 ]; then
    log "${GREEN}✓ Insample period complete${NC} (${INSAMPLE_DURATION}s)"
else
    log "${RED}✗ Insample period FAILED${NC} (exit code: $INSAMPLE_STATUS)"
    exit $INSAMPLE_STATUS
fi
echo ""

# ============================================================================
# Generate OOS Period
# ============================================================================

log_section "GENERATING OOS PERIOD (2019-2023) - AR"
log "Sampling mode: ${SAMPLING_MODE}"
log "Expected runtime: ~30 minutes"
log "Output: 2 files (180-day, 270-day AR sequences)"
echo ""

OOS_START=$(date +%s)
PYTHONPATH=. python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py \
    --period oos \
    --sampling_mode ${SAMPLING_MODE} \
    2>&1 | tee -a "$LOGFILE"
OOS_STATUS=$?
OOS_END=$(date +%s)
OOS_DURATION=$((OOS_END - OOS_START))

if [ $OOS_STATUS -eq 0 ]; then
    log "${GREEN}✓ OOS period complete${NC} (${OOS_DURATION}s)"
else
    log "${RED}✗ OOS period FAILED${NC} (exit code: $OOS_STATUS)"
    exit $OOS_STATUS
fi
echo ""

# ============================================================================
# Generate Gap Period
# ============================================================================

log_section "GENERATING GAP PERIOD (2015-2019) - AR"
log "Sampling mode: ${SAMPLING_MODE}"
log "Expected runtime: ~35 minutes"
log "Output: 2 files (180-day, 270-day AR sequences)"
echo ""

GAP_START=$(date +%s)
PYTHONPATH=. python experiments/backfill/context60/autoregressive/generate_vae_ar_sequences.py \
    --period gap \
    --sampling_mode ${SAMPLING_MODE} \
    2>&1 | tee -a "$LOGFILE"
GAP_STATUS=$?
GAP_END=$(date +%s)
GAP_DURATION=$((GAP_END - GAP_START))

if [ $GAP_STATUS -eq 0 ]; then
    log "${GREEN}✓ Gap period complete${NC} (${GAP_DURATION}s)"
else
    log "${RED}✗ Gap period FAILED${NC} (exit code: $GAP_STATUS)"
    exit $GAP_STATUS
fi
echo ""

# ============================================================================
# Validation
# ============================================================================

log_section "VALIDATING ALL GENERATED AR FILES"
log "Checking 8 files (4 periods × 2 horizons)"
echo ""

VALIDATE_START=$(date +%s)
PYTHONPATH=. python experiments/backfill/context60/autoregressive/validate_vae_ar_sequences.py \
    --sampling_mode ${SAMPLING_MODE} \
    2>&1 | tee -a "$LOGFILE"
VALIDATE_STATUS=$?
VALIDATE_END=$(date +%s)
VALIDATE_DURATION=$((VALIDATE_END - VALIDATE_START))

if [ $VALIDATE_STATUS -eq 0 ]; then
    log "${GREEN}✓ Validation complete${NC} (${VALIDATE_DURATION}s)"
else
    log "${YELLOW}⚠ Validation reported issues${NC} (exit code: $VALIDATE_STATUS)"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_DURATION / 60))
TOTAL_SEC=$((TOTAL_DURATION % 60))

log_section "AR GENERATION COMPLETE"

log "Sampling mode: ${SAMPLING_MODE}"
log ""
log "Runtime Summary:"
log "  Crisis:   ${CRISIS_DURATION}s"
log "  Insample: ${INSAMPLE_DURATION}s"
log "  OOS:      ${OOS_DURATION}s"
log "  Gap:      ${GAP_DURATION}s"
log "  Validate: ${VALIDATE_DURATION}s"
log "  TOTAL:    ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo ""

# Check output directory
OUTPUT_DIR="results/context60_baseline/predictions/autoregressive_multi_step/${SAMPLING_MODE}"
if [ -d "$OUTPUT_DIR" ]; then
    log "Output directory: $OUTPUT_DIR"
    log "Generated files:"
    ls -lh "$OUTPUT_DIR"/vae_ar_*.npz 2>&1 | tee -a "$LOGFILE"
    echo ""

    # Count files
    FILE_COUNT=$(ls "$OUTPUT_DIR"/vae_ar_*.npz 2>/dev/null | wc -l)
    log "Total files: $FILE_COUNT / 8 expected"

    # Calculate total size
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    log "Total size: $TOTAL_SIZE"
else
    log "${RED}ERROR: Output directory not found: $OUTPUT_DIR${NC}"
fi

echo ""
log "${GREEN}✓ ALL AR GENERATION COMPLETE!${NC}"
log "Sampling mode: ${SAMPLING_MODE}"
log "Logfile saved: $LOGFILE"
echo ""
log "Next steps:"
log "  1. Review validation results above"
log "  2. Compare AR vs teacher forcing results"
log "  3. Analyze error accumulation in long AR sequences"
echo ""

# Exit with validation status
exit $VALIDATE_STATUS
