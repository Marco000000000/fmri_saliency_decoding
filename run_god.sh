#!/bin/bash
#SBATCH --job-name=GOD_Eval_VC
#SBATCH --output=logs/god_eval_%A_%a.out
#SBATCH --error=logs/god_eval_%A_%a.err
#SBATCH --time=01:00:00               # Tempo ridotto: l'eval è molto più veloce
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4            # Core CPU per il dataloader
#SBATCH --mem=32G                    # RAM di sistema
#SBATCH --array=0-4                  # 5 Task totali (5 Soggetti, solo ROI VC)

# =================================================================================
# SETUP AMBIENTE E VARIABILI
# =================================================================================
SUBJECTS=(1 2 3 4 5)
ROI="VC"

# Prendo il soggetto basandomi sull'array (da 0 a 4)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================="
echo "🚀 TASK SLURM ID: ${SLURM_ARRAY_TASK_ID} | SOGGETTO: ${SUBJ} | ROI: ${ROI}"
echo "======================================================================="

# Nomi degli script e cartelle
EVAL_SCRIPT="scripts/evaluate_images.py"
BASE_OUT="results_final"
EVAL_DIR="eval_reports"
REPORT_FILE="$EVAL_DIR/S${SUBJ}_${ROI}_EVAL_ONLY_REPORT.txt"

mkdir -p logs
mkdir -p eval_reports

# Pulisco il report precedente se esiste
echo "REPORT VALUTAZIONE COMPLETA: SOGGETTO $SUBJ - ROI $ROI" > $REPORT_FILE
echo "Generato da Slurm Array Job ID: $SLURM_ARRAY_JOB_ID" >> $REPORT_FILE
echo "=======================================================================" >> $REPORT_FILE

source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

# -------------------------------------------------------------------------
# FASE UNICA: VALUTAZIONE (CICLO SULLE VARIANTI SPAZIALI)
# -------------------------------------------------------------------------
SPATIALS=("none" "mask" "box")

for SPATIAL in "${SPATIALS[@]}"; do
    echo ""
    echo ">>>>> ELABORAZIONE VARIANTE SPAZIALE: [ $SPATIAL ] <<<<<"
    
    # Intestazione nel report
    echo -e "\n\n" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE
    echo "                       VARIANTE SPAZIALE: $SPATIAL" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE

    # --- A. KANDINSKY 2-STEP EVALUATION ---
    echo "   -> Valutazione Kandinsky ($SPATIAL)..."
    echo -e "\n--- MODELLO: KANDINSKY 2-STEP | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/kandinsky/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1

    # --- B. SDXL INPAINT 2-STEP EVALUATION ---
    echo "   -> Valutazione SDXL Inpaint ($SPATIAL)..."
    echo -e "\n--- MODELLO: SDXL 2-STEP INPAINT | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/sdxl_inpaint/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1

done

echo "✅ TASK DI VALUTAZIONE COMPLETATO SOGGETTO $SUBJ - ROI $ROI."