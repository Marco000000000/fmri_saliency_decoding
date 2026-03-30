#!/bin/bash
#SBATCH --job-name=Shen_Eval_Full
#SBATCH --output=logs/shen_%j.out
#SBATCH --error=logs/shen_%j.err
#SBATCH --time=1:00:00               # Tempo massimo stimato
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4            # Core CPU per il dataloader
#SBATCH --mem=32G                    # RAM di sistema

# ==============================================================================
# Master Script: Inferenza + Valutazione (SDXL & Kandinsky) su Shen 2019
# Soggetti: 1, 2, 3
# Modalità Spaziali: none, inpaint_mask, inpaint_box
# ==============================================================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram
SUBJECTS=(1 2 3)
SPATIAL_MODES=("none" "inpaint_mask" "inpaint_box")
TEST_TYPE="natural"
ROI="ROI_VC"

# Creazione cartella per i log di valutazione e per i log di slurm (se non esiste)
LOG_DIR="evaluation_logs"
mkdir -p "$LOG_DIR"
mkdir -p logs

echo "🚀 Inizio Pipeline Completa Shen 2019 (SDXL & Kandinsky)"
echo "Soggetti: ${SUBJECTS[*]}"
echo "Modalità: ${SPATIAL_MODES[*]}"
echo "=============================================================================="

for subj in "${SUBJECTS[@]}"; do
    echo -e "\n\n======================================================="
    echo "🧠 SOGGETTO $subj"
    echo "======================================================="
    
    for mode in "${SPATIAL_MODES[@]}"; do
        
        # ---------------------------------------------------------
        # 1. Pipeline SDXL
        # ---------------------------------------------------------
        # echo -e "\n🎨 ---> Esecuzione SDXL | Modalità: $mode <---"
        # python scripts/inference_shen.py \
        #     --subject "$subj" \
        #     --test_type "$TEST_TYPE" \
        #     --spatial "$mode" \
        #     --roi "$ROI"
        
        GEN_DIR_SDXL="reconstructions/S${subj}_${TEST_TYPE}_${mode}/single"
        LOG_FILE_SDXL="${LOG_DIR}/eval_SDXL_S${subj}_${mode}.txt"
        
        if [ -d "$GEN_DIR_SDXL" ]; then
            echo "📊 Valutazione SDXL S$subj ($mode)... (salvataggio in $LOG_FILE_SDXL)"
            python scripts/evaluate_images_shen.py \
                --subject "$subj" \
                --roi "$ROI" \
                --gen_dir "$GEN_DIR_SDXL" \
                | tee "$LOG_FILE_SDXL"
        else
            echo "❌ ERRORE: Cartella SDXL $GEN_DIR_SDXL non trovata. Inferenza fallita?"
        fi

        # ---------------------------------------------------------
        # 2. Pipeline KANDINSKY
        # ---------------------------------------------------------
        # echo -e "\n🎨 ---> Esecuzione KANDINSKY | Modalità: $mode <---"
        # python scripts/inference_kandinsky_shen.py \
        #     --subject "$subj" \
        #     --test_type "$TEST_TYPE" \
        #     --spatial "$mode" \
        #     --roi "$ROI"
        
        GEN_DIR_KAND="reconstructions/kandinsky_S${subj}_${TEST_TYPE}_${mode}/single"
        LOG_FILE_KAND="${LOG_DIR}/eval_KAND_S${subj}_${mode}.txt"
        
        if [ -d "$GEN_DIR_KAND" ]; then
            echo "📊 Valutazione Kandinsky S$subj ($mode)... (salvataggio in $LOG_FILE_KAND)"
            python scripts/evaluate_images_shen.py \
                --subject "$subj" \
                --roi "$ROI" \
                --gen_dir "$GEN_DIR_KAND" \
                | tee "$LOG_FILE_KAND"
        else
            echo "❌ ERRORE: Cartella Kandinsky $GEN_DIR_KAND non trovata. Inferenza fallita?"
        fi
        
    done
done

echo -e "\n🎉 =============================================================================="
echo "✅ TUTTI GLI ESPERIMENTI SONO TERMINATI!"
echo "Tutte le metriche (LPIPS, CLIP, AlexNet, Mask IoU) sono state salvate in: $LOG_DIR/"
echo "=============================================================================="