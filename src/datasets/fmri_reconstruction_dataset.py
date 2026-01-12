import os
from typing import Dict, Iterable, List, Sequence, Tuple

import bdpy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_fmri_image_pairs(
    embedding_dir: str,
    image_dir: str,
    h5_file: str,
    rois: Dict[str, str],
    img_extensions: Iterable[str] = (".png",".JPEG"),
) -> Dict[str, List]:
    """
    Load fmri/image pairs for a subject file, concatenating the requested ROIs.

    Args:
        embedding_dir: Directory that contains the fmri h5 file.
        image_dir: Directory containing the corresponding images.
        h5_file: Name of the h5 file to read.
        rois: Dict mapping ROI names to bdpy selector strings.
        img_extensions: Allowed image extensions (order is used to resolve collisions).

    Returns:
        Dict with fmri arrays, image paths and the class-to-index mapping.
        Returns None if the h5 file is missing or no ROI data is found.
    """
    subject_h5_file_path = os.path.join(embedding_dir, h5_file)
    fmri: List[np.ndarray] = []
    image_paths: List[str] = []

    if not os.path.exists(subject_h5_file_path):
        print(f"Warning: H5 file not found: {subject_h5_file_path}")
        return None

    dat = bdpy.BData(subject_h5_file_path)
    x_concatenated_roi_data = None
    active_roi_bdpy_selectors = [val for val in rois.values()]

    if not active_roi_bdpy_selectors:
        print(f"Warning: No active ROI specified for {subject_h5_file_path} with config: {rois}.")
        return None

    for roi_selector_str_for_bdpy in active_roi_bdpy_selectors:
        try:
            current_roi_data = dat.select(roi_selector_str_for_bdpy)
            if x_concatenated_roi_data is None:
                x_concatenated_roi_data = current_roi_data
            else:
                x_concatenated_roi_data = np.concatenate((x_concatenated_roi_data, current_roi_data), axis=1)
        except Exception as e:
            print(f"Warning: ROI '{roi_selector_str_for_bdpy}' not found in {subject_h5_file_path}: {e}")
            continue

    if x_concatenated_roi_data is None:
        print(f"Warning: No fMRI data loaded from {subject_h5_file_path} with ROIs {rois}.")
        return None

    stimulus_labels_from_h5 = dat.get_labels("stimulus_name")
    classes_index: Dict[str, List[int]] = {}
    for i, image_stimulus_key in enumerate(stimulus_labels_from_h5):
        candidate_paths = [
            os.path.join(image_dir, image_stimulus_key + ext) for ext in img_extensions
        ]
        image_file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if image_file_path is None:
            print(f"Warning: Image not found for {image_stimulus_key} with extensions {img_extensions}. Skipping.")
            continue

        fmri.append(x_concatenated_roi_data[i])
        current_idx = len(fmri) - 1
        class_key = image_stimulus_key.split("_")[0]
        if class_key not in classes_index:
            classes_index[class_key] = []
        classes_index[class_key].append(current_idx)
        image_paths.append(image_file_path)

    return {"fmri": fmri, "image_paths": image_paths, "classes_index": classes_index}


def split_data_for_class(data_dict: Dict[str, List], train_ratio: float = 0.9) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Split fmri/image examples by class to build train/val splits.
    """
    classes_index = data_dict["classes_index"]
    fmri = np.array(data_dict["fmri"])
    image_paths = np.array(data_dict["image_paths"])

    all_classes = list(classes_index.keys())
    np.random.shuffle(all_classes)

    num_classes_for_train = int(len(all_classes) * train_ratio)
    classes_for_train = all_classes[:num_classes_for_train]
    classes_for_val = all_classes[num_classes_for_train:]

    if not classes_for_val:
        if len(classes_for_train) > 1:
            classes_for_val = [classes_for_train.pop()]
        else:
            print("Warning: Not enough classes for a separate validation set; using training set also for validation.")
            classes_for_val = classes_for_train

    train_indices = [idx for cls in classes_for_train for idx in classes_index[cls]]
    val_indices = [idx for cls in classes_for_val for idx in classes_index[cls]]

    train_dict = {
        "fmri": [fmri[i] for i in train_indices],
        "image_paths": [image_paths[i] for i in train_indices],
    }
    val_dict = {
        "fmri": [fmri[i] for i in val_indices],
        "image_paths": [image_paths[i] for i in val_indices],
    }

    return train_dict, val_dict


class FmriImageDataset(Dataset):
    """
    Minimal Dataset wrapping pre-loaded fmri vectors and image paths.
    """

    def __init__(self, fmri: Sequence, image_paths: Sequence[str], transform=None, load_img: bool = False):
        self.fmri = fmri
        self.image_paths = image_paths
        self.transform = transform if transform else transforms.ToTensor()
        self.load_img = load_img
        self.images: List[Image.Image] = []
        if self.load_img:
            for image_path in self.image_paths:
                try:
                    self.images.append(Image.open(image_path).convert("L"))
                except FileNotFoundError:
                    print(f"Warning: Image not found: {image_path}. It will be skipped.")

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        fmri_tensor = torch.from_numpy(self.fmri[idx]).float().squeeze(0)
        image = self.images[idx] if self.load_img else Image.open(self.image_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)

        image_sum = image.sum()
        if image_sum > 0:
            image = image / image_sum

        return fmri_tensor, image