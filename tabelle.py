import os
import re
import numpy as np
import warnings

# Disabilitiamo i warning per le medie calcolate su array vuoti
warnings.filterwarnings('ignore', r'Mean of empty slice')
warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')

# ================= CONFIGURAZIONE =================
SUBJECTS = [1, 2, 3, 4, 5]
# Aggiungi qui tutte le ROI che hai analizzato (es. V1, V2, V3, FFA, PPA, LOC, VC)
ROIS = ["VC", "V1", "V2", "V3"] 
SPACES = ["none", "mask", "box"]

REPORT_DIR = "eval_reports"

MODELS_CONFIG = {
    "Kamitani Baseline": {"search_id": "KAMITANI 2019 BASELINE", "file_tmpl": "KAMITANI_S{sub}_{roi}_REPORT.txt"},
    "Kandinsky 2-Step": {"search_id": "KANDINSKY 2-STEP", "file_tmpl": "S{sub}_{roi}_FULL_REPORT.txt"},
    "SDXL 2-Step": {"search_id": "SDXL 2-STEP INPAINT", "file_tmpl": "S{sub}_{roi}_FULL_REPORT.txt"}
}
# ==================================================

METRICS_DICT = {
    "1. Pure Semantic": r"1\. PURE SEMANTIC.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%",
    "2. Pure Spatial": r"2\. PURE SPATIAL.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%",
    "3. Pure Image (CLIP)": r"3\. PURE IMAGE - CLIP.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%",
    "4. Pure Image (AlexNet)": r"4\. PURE IMAGE - ALEXNET.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%",
    "5. Unbiased (CLIP)": r"5\. UNBIASED GEN CROSS-MASKED - CLIP.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%",
    "6. Unbiased (AlexNet)": r"6\. UNBIASED GEN CROSS-MASKED - ALEXNET.*?\|\s*([\d.]+)%\s*\|\s*([\d.]+)%"
}

def parse_report(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        content = f.read()
    sections = re.split(r'--- MODELLO: (.*?) \| SPAZIO: (.*?) ---', content)
    parsed_data = {}
    for i in range(1, len(sections), 3):
        model = sections[i].strip()
        space = sections[i+1].strip()
        text = sections[i+2]
        if model not in parsed_data:
            parsed_data[model] = {}
        parsed_data[model][space] = text
    return parsed_data

def extract_scores(text_block):
    scores = {}
    for metric_name, pattern in METRICS_DICT.items():
        match = re.search(pattern, text_block)
        if match:
            scores[metric_name] = {"Top-1": float(match.group(1)), "Top-5": float(match.group(2))}
        else:
            scores[metric_name] = {"Top-1": np.nan, "Top-5": np.nan}
    return scores

def main():
    print(f"📊 GENERAZIONE TABELLE MULTI-ROI E MULTI-SPAZIO")
    print(f"📂 Cerco i log nella cartella: {REPORT_DIR}/")
    
    # Struttura dati: data[roi][space][metric][top_k][model] = [s1, s2, s3, s4, s5]
    all_data = {
        r: {s: {m: {k: {mod: [] for mod in MODELS_CONFIG} for k in ["Top-1", "Top-5"]} for m in METRICS_DICT} for s in SPACES}
        for r in ROIS
    }
    
    # Popoliamo i dati
    for roi in ROIS:
        for sub in SUBJECTS:
            file_cache = {}
            for model_display_name, config in MODELS_CONFIG.items():
                filepath = os.path.join(REPORT_DIR, config["file_tmpl"].format(sub=sub, roi=roi))
                search_id = config["search_id"]
                
                if filepath not in file_cache:
                    file_cache[filepath] = parse_report(filepath)
                parsed_data = file_cache[filepath]
                
                for space in SPACES:
                    text_block = parsed_data.get(search_id, {}).get(space, "")
                    scores = extract_scores(text_block) if text_block else extract_scores("")
                    
                    for m in METRICS_DICT:
                        for k in ["Top-1", "Top-5"]:
                            all_data[roi][space][m][k][model_display_name].append(scores[m][k])

    def format_row(name, values):
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        v_str = [f"{v:05.2f}%" if not np.isnan(v) else " N/A  " for v in values]
        mean_str = f"{mean_val:05.2f}% ± {std_val:05.2f}%" if not np.isnan(mean_val) else "N/A"
        return f"{name:<20} | {v_str[0]:<7} | {v_str[1]:<7} | {v_str[2]:<7} | {v_str[3]:<7} | {v_str[4]:<7} | {mean_str}"

    # Stampa i risultati
    for roi in ROIS:
        print("\n\n" + "#"*110)
        print(f"####################################  RISULTATI ROI: {roi.upper()}  ####################################")
        print("#"*110)
        
        for space in SPACES:
            print("\n" + "="*100)
            print(f"============================ SPAZIO: {space.upper():<6} ============================")
            print(f"{'METRICA & MODELLO':<20} | {'S1':<7} | {'S2':<7} | {'S3':<7} | {'S4':<7} | {'S5':<7} | MEDIA ± STD")
            print("-" * 100)
            
            for m in METRICS_DICT:
                # La Pure Spatial non esiste per lo spazio "none" e non ha senso calcolarla
                if space == "none" and m == "2. Pure Spatial":
                    continue
                    
                for k in ["Top-1", "Top-5"]:
                    print(f"--- {m} ({k}) ---")
                    for model_display_name in MODELS_CONFIG:
                        vals = all_data[roi][space][m][k][model_display_name]
                        print(format_row(model_display_name, vals))
                print("-" * 100)

if __name__ == "__main__":
    main()