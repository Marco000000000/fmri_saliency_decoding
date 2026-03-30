#!/bin/bash
#SBATCH --job-name=GOD_Ablation_Full
#SBATCH --output=logs/god_%A_%a.out
#SBATCH --error=logs/god_%A_%a.err
#SBATCH --time=24:00:00               # Tempo massimo stimato per 1 Soggetto/1 ROI (adattalo se serve)
#SBATCH --job-name=my_training_job
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4            # Core CPU per il dataloader
#SBATCH --mem=32G                    # RAM di sistema
#SBATCH --array=0-39                 # 40 Task totali (5 Soggetti x 8 ROI)

# =================================================================================
# SETUP AMBIENTE (Scommenta e adatta al tuo cluster)
# =================================================================================
# module load anaconda3/2023.03
# module load cuda/11.8
# source activate fMRI_env

# =================================================================================
# MAPPATURA ARRAY ID -> SOGGETTO E ROI
# =================================================================================
SUBJECTS=(1 2 3 4 5)
ROIS=("V1" "V2" "V3" "V4" "LOC" "FFA" "PPA" "VC")

# Calcolo l'indice del soggetto (0-4) e della ROI (0-7) basandomi sul TASK_ID (0-39)
SUBJ_IDX=$(( SLURM_ARRAY_TASK_ID / 8 ))
ROI_IDX=$(( SLURM_ARRAY_TASK_ID % 8 ))

SUBJ=${SUBJECTS[$SUBJ_IDX]}
ROI=${ROIS[$ROI_IDX]}

echo "======================================================================="
echo "🚀 TASK SLURM ID: ${SLURM_ARRAY_TASK_ID} | SOGGETTO: ${SUBJ} | ROI: ${ROI}"
echo "======================================================================="

# Nomi degli script
TRAIN_STRUCT="scripts/train_depth_decoder.py"
TRAIN_SEMANTIC="scripts/train_clip_no_adam.py"
GEN_KANDINSKY="scripts/inference_clip_no_adam_seg.py"             
GEN_SDXL_INPAINT="scripts/inference_sdxl.py"       
GEN_SDXL_1STEP="scripts/inference_box1.py"       
EVAL_SCRIPT="scripts/evaluate_images.py"

BASE_OUT="results_final"
EVAL_DIR="eval_reports"
REPORT_FILE="$EVAL_DIR/S${SUBJ}_${ROI}_FULL_REPORT.txt"
mkdir -p results_final
mkdir -p logs
mkdir -p eval_reports
# Pulisco il report precedente se esiste
echo "REPORT VALUTAZIONE COMPLETA: SOGGETTO $SUBJ - ROI $ROI" > $REPORT_FILE
echo "Generato da Slurm Array Job ID: $SLURM_ARRAY_JOB_ID" >> $REPORT_FILE
echo "=======================================================================" >> $REPORT_FILE
source $(conda info --base)/etc/profile.d/conda.sh

# 2. Attiva l'ambiente (Questa è la tua "concatenazione")
conda activate labram
#  -------------------------------------------------------------------------
# FASE 1: TRAINING
# -------------------------------------------------------------------------
echo "[1/3] 🏋️‍♂️ Addestramento Modelli fMRI per S${SUBJ} ${ROI}..."
python $TRAIN_STRUCT --subject $SUBJ --rois $ROI
python $TRAIN_SEMANTIC --subject $SUBJ --rois $ROI

# -------------------------------------------------------------------------
# FASE 2 & 3: GENERAZIONE E VALUTAZIONE (CICLO SULLE VARIANTI SPAZIALI)
# -------------------------------------------------------------------------
SPATIALS=("none" "mask" "box")

for SPATIAL in "${SPATIALS[@]}"; do
    echo ""
    echo ">>>>> ELABORAZIONE VARIANTE SPAZIALE: [ $SPATIAL ] <<<<<"
    
    # Intestazione nel report
    echo -e "\n\n" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE
    echo "                      VARIANTE SPAZIALE: $SPATIAL" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE

    # --- A. KANDINSKY 2-STEP ---
    echo "   -> Generazione Kandinsky ($SPATIAL)..."
    python $GEN_KANDINSKY --subject $SUBJ --rois $ROI --spatial $SPATIAL --guidance_scale 1.5 \
        --output_dir "$BASE_OUT/kandinsky"
    
    echo "   -> Valutazione Kandinsky ($SPATIAL)..."
    echo -e "\n--- MODELLO: KANDINSKY 2-STEP | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/kandinsky/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1


    # --- B. SDXL INPAINT 2-STEP ---
    echo "   -> Generazione SDXL Inpaint ($SPATIAL)..."
    python $GEN_SDXL_INPAINT --subject $SUBJ --rois $ROI --spatial $SPATIAL --guidance_scale 3.5 \
        --output_dir "$BASE_OUT/sdxl_inpaint"
    
    echo "   -> Valutazione SDXL Inpaint ($SPATIAL)..."
    echo -e "\n--- MODELLO: SDXL 2-STEP INPAINT | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
        --gen_dir "$BASE_OUT/sdxl_inpaint/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1


    # # --- C. SDXL ATTENTION 1-STEP ---
    # echo "   -> Generazione SDXL Attention ($SPATIAL)..."
    # python $GEN_SDXL_1STEP --subject $SUBJ --rois $ROI --spatial $SPATIAL --guidance_scale 4.5 \
    #     --output_dir "$BASE_OUT/sdxl_attention"
    
    # echo "   -> Valutazione SDXL Attention ($SPATIAL)..."
    # echo -e "\n--- MODELLO: SDXL 1-STEP ATTENTION | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    # python $EVAL_SCRIPT --subject $SUBJ --rois $ROI  \
    #     --gen_dir "$BASE_OUT/sdxl_attention/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1

done

echo "✅ TASK COMPLETATO SOGGETTO $SUBJ - ROI $ROI."