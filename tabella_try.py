import os
import re
import glob
import pandas as pd

def parse_ablation_logs():
    data = []

    print("🔍 Ricerca log di ablazione in corso...")

    # ==========================================
    # 1. Parsing dei log di GOD
    # ==========================================
    god_dir = "eval_reports"
    if os.path.exists(god_dir):
        god_files = glob.glob(f"{god_dir}/S*_ABLATION_REPORT.txt")
        print(f" -> Trovati {len(god_files)} file per GOD.")
        
        for filepath in god_files:
            subj_match = re.search(r'S(\d+)_', os.path.basename(filepath))
            if not subj_match: continue
            subject = subj_match.group(1)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            spatial, eval_mode, model = "unknown", "unknown", "unknown"
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Macchina a stati per leggere i blocchi del file Bash di GOD
                if "VARIANTE SPAZIALE GENERAZIONE:" in line:
                    spatial = line.split(":")[-1].strip()
                elif "---> Valutazione Metrica:" in line:
                    eval_mode = re.search(r'Metrica:\s*([A-Z]+)', line).group(1).lower()
                elif line.startswith(">> "):
                    # Estrae il nome del modello (es: KANDINSKY 2-STEP)
                    model = line.replace(">>", "").split("(")[0].strip()
                elif line.startswith("--- RISULTATI"):
                    # Trovato blocco risultati, leggiamo le metriche successive
                    res = {
                        "Dataset": "GOD", "Subject": f"S{subject}", 
                        "Model": model, "Spatial_Variant": spatial, "Eval_Mode": eval_mode
                    }
                    i += 1
                    while i < len(lines) and lines[i].strip() != "" and ":" in lines[i]:
                        k, v = lines[i].strip().split(":", 1)
                        # Pulizia nomi metriche
                        k = k.replace(" ↓", "").strip() 
                        res[k] = v.strip()
                        i += 1
                    data.append(res)
                i += 1

    # ==========================================
    # 2. Parsing dei log di SHEN
    # ==========================================
    shen_dir = "evaluation_logs_ablation"
    if os.path.exists(shen_dir):
        shen_files = glob.glob(f"{shen_dir}/*.txt")
        print(f" -> Trovati {len(shen_files)} file per SHEN.")
        
        for filepath in shen_files:
            filename = os.path.basename(filepath)
            # Regex per estrarre info dal nome file di Shen
            m = re.search(r'eval_(SDXL|KAND)_S(\d+)_spatial-(.*)_eval-(.*)\.txt', filename)
            if not m: continue
            
            model_abbr, subject, spatial, eval_mode = m.groups()
            model_name = "SDXL 2-STEP INPAINT" if model_abbr == "SDXL" else "KANDINSKY 2-STEP"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Cerca il blocco "--- RISULTATI [MODE] ---" e prende il testo fino alla riga vuota
            res_match = re.search(r'--- RISULTATI.*?\n(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if res_match:
                res = {
                    "Dataset": "SHEN", "Subject": f"S{subject}", 
                    "Model": model_name, "Spatial_Variant": spatial, "Eval_Mode": eval_mode
                }
                for l in res_match.group(1).strip().split('\n'):
                    if ":" in l:
                        k, v = l.split(":", 1)
                        k = k.replace(" ↓", "").strip()
                        res[k] = v.strip()
                data.append(res)

    # ==========================================
    # 3. Creazione ed Export DataFrame
    # ==========================================
    if not data:
        print("❌ Nessun dato estratto. Controlla che i file log non siano vuoti o falliti.")
        return

    df = pd.DataFrame(data)
    
    # Riorganizziamo l'ordine delle colonne in modo logico
    cols_order = ['Dataset', 'Subject', 'Model', 'Spatial_Variant', 'Eval_Mode', 
                  'LPIPS', 'CLIP-B', 'CLIP-XL', 'Alex2', 'Alex5', 'Alex7']
    # Assicuriamoci di prendere solo le colonne che esistono effettivamente
    cols = [c for c in cols_order if c in df.columns] + [c for c in df.columns if c not in cols_order]
    df = df[cols]

    # Ordiniamo le righe per avere una lettura facile
    df = df.sort_values(by=['Dataset', 'Eval_Mode', 'Spatial_Variant', 'Model', 'Subject'])

    out_name = "Ablation_Generative_Results.csv"
    df.to_csv(out_name, index=False)
    print(f"\n🎉 BOOM! Dati aggregati salvati con successo in: {out_name}")
    print(f"   -> Totale righe estratte: {len(df)}")

if __name__ == "__main__":
    parse_ablation_logs()