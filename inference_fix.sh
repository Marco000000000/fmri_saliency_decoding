#!/bin/bash
#SBATCH --job-name=Fix_GOD
#SBATCH --output=logs/fix_%A_%a.out
#SBATCH --error=logs/fix_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=train
#SBATCH --array=0-39                 # 40 Task totali

source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

SUBJECTS=(1 2 3 4 5)
ROIS=("V1" "V2" "V3" "V4" "LOC" "FFA" "PPA" "VC")

SUBJ_IDX=$(( SLURM_ARRAY_TASK_ID / 8 ))
ROI_IDX=$(( SLURM_ARRAY_TASK_ID % 8 ))

SUBJ=${SUBJECTS[$SUBJ_IDX]}
ROI=${ROIS[$ROI_IDX]}

echo "======================================================================="
echo "📊 RIVALUTAZIONE (PESI MSE): SOGGETTO ${SUBJ} | ROI: ${ROI}"
echo "======================================================================="

EVAL_SCRIPT="scripts/evaluate_images.py"
BASE_OUT="results_final"
EVAL_DIR="eval_reports"
REPORT_FILE="$EVAL_DIR/S${SUBJ}_${ROI}_FULL_REPORT.txt"

# Inizializza il report
echo "REPORT VALUTAZIONE COMPLETA (MSE FIXED): SOGGETTO $SUBJ - ROI $ROI" > $REPORT_FILE
echo "=======================================================================" >> $REPORT_FILE

SPATIALS=("none" "mask" "box")

for SPATIAL in "${SPATIALS[@]}"; do
    echo -e "\n\n" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE
    echo "                      VARIANTE SPAZIALE: $SPATIAL" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE

    # --- A. KANDINSKY 2-STEP (Solo Valutazione) ---
    echo "   -> Valutazione Kandinsky ($SPATIAL)..."
    echo -e "\n--- MODELLO: KANDINSKY 2-STEP | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/kandinsky/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1

    # --- B. SDXL INPAINT 2-STEP (Solo Valutazione) ---
    echo "   -> Valutazione SDXL Inpaint ($SPATIAL)..."
    echo -e "\n--- MODELLO: SDXL 2-STEP INPAINT | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/sdxl_inpaint/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1

    # --- C. SDXL ATTENTION 1-STEP (LANCIA SOLO QUANDO SPATIAL = BOX) ---
    if [ "$SPATIAL" == "box" ]; then
        echo "   -> RIGENERAZIONE SDXL Attention (box)..."
        # Rimuoviamo  dal comando python visto che lo script non lo usa
        python scripts/inference_box1.py --subject $SUBJ --rois $ROI --guidance_scale 4.5 \
            --output_dir "$BASE_OUT/sdxl_attention"
        
        echo "   -> Valutazione SDXL Attention (box)..."
        echo -e "\n--- MODELLO: SDXL 1-STEP ATTENTION | SPAZIO: box ---" >> $REPORT_FILE
        python $EVAL_SCRIPT --subject $SUBJ --rois $ROI --spatial box \
            --gen_dir "$BASE_OUT/sdxl_attention/S${SUBJ}_${ROI}_box/single" >> $REPORT_FILE 2>&1
    fi

done

echo "✅ Completato S${SUBJ} - ${ROI}."