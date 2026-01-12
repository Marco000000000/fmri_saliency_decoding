import argparse
import json

# Mapping between ROI keys and bdpy selector strings
ALL_ROI_MAPPINGS = {
    "VC": "ROI_VC = 1",
    "V1": "ROI_V1 = 1",
    "V2": "ROI_V2 = 1",
    "V3": "ROI_V3 = 1",
    "V4": "ROI_V4 = 1",
    "LOC": "ROI_LOC = 1",
    "FFA": "ROI_FFA = 1",
    "PPA": "ROI_PPA = 1",
}

# fMRI input dimensions per subject for each ROI
subject_dims = {
    "VC": [3444, 4979, 5355, 4656, 6237],
    "V1": [826, 751, 918, 925, 962],
    "V2": [857, 766, 801, 927, 1440],
    "V3": [688, 632, 690, 747, 1433],
    "V4": [266, 333, 381, 453, 1156],
    "LOC": [470, 1511, 1678, 653, 833],
    "FFA": [238, 137, 268, 196, 1310],
    "PPA": [144, 332, 585, 357, 757],
}


def str_to_bool(val):
    if isinstance(val, bool):
        return val
    val_str = str(val).lower()
    if val_str in {"true", "1", "yes", "y"}:
        return True
    if val_str in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Valore booleano non valido: {val}")


def parse_roi_keys(raw_rois):
    try:
        parsed = json.loads(raw_rois)
        if isinstance(parsed, str):
            return [parsed]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [roi.strip() for roi in raw_rois.split(",") if roi.strip()]


def parse_image_exts(raw_exts):
    if isinstance(raw_exts, (list, tuple)):
        return tuple(raw_exts)
    return tuple(ext.strip() for ext in str(raw_exts).split(",") if ext.strip())