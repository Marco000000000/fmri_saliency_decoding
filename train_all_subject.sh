#!/bin/bash

# ==============================================================================
# SCRIPT AUTOMATICO PER TRAINING E INFERENCE SU TUTTI I SOGGETTI E ROI
# ==============================================================================

# Definisci la lista dei soggetti (da 1 a 5, basato sulla lunghezza dei tuoi subject_dims)
SUBJECTS=( 3 4 5)

# Definisci la lista delle ROI (Le chiavi del dizionario ALL_ROI_MAPPINGS)
ROIS=("VC" "V1" "V2" "V3" "V4" "LOC" "FFA" "PPA")

# Path Python (opzionale: se usi un venv specifico, sostituisci "python" col path completo)
PYTHON_CMD="python"

echo "=================================================="
echo "AVVIO ESPERIMENTI BATCH"
echo "Soggetti: ${SUBJECTS[*]}"
echo "ROIs: ${ROIS[*]}"
echo "=================================================="

# Loop sui Soggetti
for SUB in "${SUBJECTS[@]}"; do
    
    # Loop sulle ROI
    for ROI in "${ROIS[@]}"; do
        
        echo ""
        echo "--------------------------------------------------"
        echo ">>> ELABORAZIONE: Subject $SUB | ROI $ROI"
        echo "--------------------------------------------------"

        # 1. TRAINING
        echo "[1/2] Avvio Training..."
        $PYTHON_CMD scripts/train_clip_no_adam.py --subject "$SUB" --rois "$ROI" --epochs 20

        # Verifica codice di uscita del training (0 = successo)
        if [ $? -eq 0 ]; then
            echo "[OK] Training completato."
            
            # 2. INFERENCE
            echo "[2/2] Avvio Inference..."
            $PYTHON_CMD scripts/inference_clip_no_adam_2.py --subject "$SUB" --rois "$ROI"
            
            if [ $? -eq 0 ]; then
                echo "[OK] Inference completata per S$SUB - $ROI."
            else
                echo "[ERR] Errore durante l'Inference per S$SUB - $ROI."
            fi

        else
            echo "[ERR] Training fallito per S$SUB - $ROI. Salto l'Inference."
        fi

    done
done

echo ""
echo "=================================================="
echo "TUTTI I JOB COMPLETATI"
echo "=================================================="