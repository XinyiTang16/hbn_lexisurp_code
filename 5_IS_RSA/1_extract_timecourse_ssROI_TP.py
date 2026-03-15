"""
Extract ROI time series and phase angles for movie-watching fMRI data (TP).

This script extracts ROI mean time series of TP from the top 10% of voxels within each ROI based on subject-specific lexical surprisal encoding maps in DM. 
It then computes phase angles using the Hilbert transform to prepare for inter-subject phase synchrony (ISPS) analyses.

Outputs:
- ROI mean time series (top 10% voxels per subject)
- Phase angle time series
"""
import numpy as np
import nibabel as nib
import os
from glob import glob
from collections import defaultdict
from scipy.signal import hilbert
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path

# ---------------------------- Paths and Setup ----------------------------
DATA_dir = 'movieTP_subdata/'
subjects = sorted([f.split('_')[0].replace("sub-","") for f in os.listdir(DATA_dir) if f.endswith('.hdf')])
print("Total number of subjects:", len(subjects))
print("First 10 subjects:", subjects[:10])  # Print first 10 subjects for verification

# Use GPT3_surprisal map from movie DM to define ssROIs
CON = "GPT3_surprisal"
ROOT_FOLDER = Path("movieDM_lexisurp_10feature_result/")
pattern = ROOT_FOLDER / "*NDAR*" / f"*{CON}*.nii.gz"
contrast_files = sorted(glob(str(pattern)))
contrast_map_dict = {f.split("/")[-1].split("_")[0].replace("sub-",""): f for f in contrast_files}
print("Total number of contrast map:", len(contrast_map_dict))
print("First 10 contrast maps:", list(contrast_map_dict.keys())[:10])

brain_mask = nib.load('MNI152_T1_1mm_brain_resampled_HBN_brainmask.nii.gz').get_fdata().flatten() > 0
brain_to_flat_idx = np.where(brain_mask)[0]
mask_index_map = {idx: i for i, idx in enumerate(brain_to_flat_idx)}

ROI_dir = '~/2_lexisurp_selectivity/unthres_ROI/'
ROIs = sorted(glob(os.path.join(ROI_dir, 'rROI*CoreLanguage*.nii')))
ROI_dict = {"_".join(os.path.splitext(f)[0].split('_')[-2:]): f for f in ROIs}

# ----------------------- Build Network Dictionary ------------------------
network_dict = defaultdict(list)
for roi_path in ROIs:
    fname = os.path.basename(roi_path).replace('.nii', '')
    parts = fname.split("_", 2)
    _, net, roi = parts
    full_name = f"{net}_{roi}"
    network_dict[f"Whole_{net}"].append(full_name)
print("Network dictionary:",network_dict)

# ------------------------- Helper Functions -----------------------------
def load_subject_data(subject_id):
    fpath = os.path.join(DATA_dir, f"sub-{subject_id}_movieTP_wholebrain.hdf")
    with h5py.File(fpath, 'r') as f:
        return f["fmri_response"][:].astype(np.float32)  # (T, V)

def extract_top_voxel_indices_from_DM_contrast(subject_contrast_path, roi_path):
    # uses movieDM contrast to select voxels, but extracts time series from movieTP data
    contrast_img = nib.load(subject_contrast_path).get_fdata()
    roi_mask = nib.load(roi_path).get_fdata() > 0
    masked_values = contrast_img[roi_mask]
    valid_mask = (~np.isnan(masked_values))
    valid_values = masked_values[valid_mask]
    
    if len(valid_values) == 0:
        return np.array([], dtype=int)
    
    n_top_10_perc = int(np.ceil(len(valid_values) * 0.1))  # Compute the number of voxels in the top 10%
    n = n_top_10_perc
    #print(roi_path,n)
    
    top_indices = np.argsort(valid_values)[-n:] if len(valid_values) >= n else np.argsort(valid_values)
    all_voxel_indices = np.where(roi_mask.flatten())[0]
    selected_flat_indices = all_voxel_indices[valid_mask][top_indices]
    valid_indices = [mask_index_map[i] for i in selected_flat_indices if i in mask_index_map]
    return np.array(valid_indices, dtype=int)

def compute_top_voxel_ts_parallel(subjects, roi_names, n_jobs):
    
    sample_data = load_subject_data(subjects[0])
    T, V = sample_data.shape
    n_rois = len(roi_names)
    n_subs = len(subjects)

    def process_subject(s_idx, sub):
        
        subj_data = load_subject_data(sub)
        contrast_path = contrast_map_dict.get(sub)
        
        if contrast_path is None:
            raise ValueError(f"No contrast map found for subject {sub}")
        
        subj_ts = np.zeros((T, n_rois))
        
        for r_idx, roi in enumerate(roi_names):

            voxels = extract_top_voxel_indices_from_DM_contrast(contrast_path, ROI_dict[roi])

            if len(voxels) == 0:
                subj_ts[:, r_idx] = np.nan
            else:
                subj_ts[:, r_idx] = np.nanmean(subj_data[:, voxels], axis=1)
        return s_idx, subj_ts

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(s_idx, sub) for s_idx, sub in enumerate(tqdm(subjects, desc="Extracting top-100 voxel TS"))
    )

    ts_data = np.zeros((T, n_rois, n_subs))
    
    for s_idx, subj_ts in results:
        ts_data[:, :, s_idx] = subj_ts

    return ts_data # (T, R, S)

def compute_phase_angles(ts_data):
    return np.angle(hilbert(ts_data, axis=0))

# ------------------------------ Run Pipeline -----------------------------
out_dir = "5_movieTP_ISRSA/1_sstop10perc_timecourse/"
os.makedirs(out_dir, exist_ok=True)

for net_name, roi_list in network_dict.items():

    print(f"\nProcessing {net_name} with {len(roi_list)} ROIs")
    net_out_dir = os.path.join(out_dir, net_name)
    os.makedirs(net_out_dir, exist_ok=True)
    
    sstop_ts_data = compute_top_voxel_ts_parallel(subjects, roi_list, n_jobs=10) 
    print("ts.shape", sstop_ts_data.shape) # (T, R, S)
    np.save(os.path.join(net_out_dir, f'movieTP_{net_name}_sstop10perc_DMdefined_ROI_meantimecourse.npy'), sstop_ts_data)
    
    phase_data = compute_phase_angles(sstop_ts_data)
    T, R, S = phase_data.shape
    print("phase.shape", phase_data.shape) # (T, R, S)
    np.save(os.path.join(net_out_dir, f'movieTP_{net_name}_sstop10perc_DMdefined_ROI_meantimecourse_phase.npy'), phase_data)
