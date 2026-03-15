"""
This script uses cross-movie generalization to test how well 10 movie features trained on DM predict movie-watching fMRI responses TP for each subject.
It saves whole-model and feature-specific brain maps of Fisher z-transformed correlation scores.

Inputs:
- Movie feature regressors in HDF5 format
- Subject fMRI response data in HDF5 format
- Brain mask NIfTI file

Outputs:
- NIfTI maps and glass-brain plots of mean Fisher z-transformed scores

Notes:
- Uses multiple-kernel ridge regression with delayed features
- Uses the 'Delayer' class from the Gallant Lab voxelwise modeling tutorials: https://gallantlab.org/voxelwise_tutorials/pages/index.html
- Uses GPU acceleration if available
"""
import os
import h5py
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import correlation_score, correlation_score_split
from himalaya.backend import set_backend
from delayer import Delayer
import pandas as pd

# Set configurations
set_config(display='diagram')
backend = set_backend("cupy", on_error="warn")
print("Backend:", backend.to_numpy.__module__)

# Load Feature Data (Predictors)
def load_feature_data(filepath):
    with h5py.File(filepath, 'r') as f:
        return f["features"][:].T  # Shape: (250, n_features)

X_train = load_feature_data('regressors/movieDM_lexisurp_10features_lanczos.hdf')
X_train = X_train.astype("float32")
print("X_train", X_train.shape)

X_test = load_feature_data('regressors/movieTP_lexisurp_10features_lanczos.hdf')
X_test = X_test.astype("float32")
print("X_test", X_test.shape)

CONS = ["GPT3_surprisal", "Lg10WF", "spoken_words", "written_words", 
        "positive", "negative", "faces", "body", "brightness", "loudness"]

# Load whole-brain mask (resampled MNI152 template) used to map voxelwise scores back to 3D brain space
MASK_FILE = 'MNI152_T1_1mm_brain_resampled_HBN_brainmask.nii.gz'
mask_img = nib.load(MASK_FILE)
gray_mask = mask_img.get_fdata()
space_size = np.prod(gray_mask.shape)

# Define MKRR Model Pipeline
def create_pipeline():
    alphas = np.logspace(1, 20, 20)
    solver_params = dict(n_iter=20, alphas=alphas, n_targets_batch=200, 
                         n_alphas_batch=5, n_targets_batch_refit=200)
    
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search", 
                                      solver_params=solver_params, cv=10)
    
    preprocess_pipeline = make_pipeline(
        Delayer(delays=range(1, 13)),  # Apply time delays
        Kernelizer(kernel="linear")   # Apply linear kernel
    )
    
    feature_slices = [slice(i, i + 1) for i in range(X_train.shape[1])]
    kernelizers = [(name, preprocess_pipeline, slice_) for name, slice_ in zip(CONS, feature_slices)]
    column_kernelizer = ColumnKernelizer(kernelizers)
    
    return make_pipeline(column_kernelizer, mkr_model)

create_pipeline()

# Function to process and save results
def save_nifti_and_plot(scores, title, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_nii = os.path.join(output_dir, f"{title}.nii.gz")
    output_png = os.path.join(output_dir, f"{title}.png")
    
    data_mask = np.zeros(space_size)
    inds = np.where(gray_mask.reshape(space_size) > 0)
    data_mask[inds] = scores
    data_mask = data_mask.reshape(gray_mask.shape)
    nii_img = nib.Nifti1Image(data_mask, mask_img.affine, mask_img.header)
    nii_img.to_filename(output_nii)
    
    plotting.plot_glass_brain(nii_img, title=title, display_mode='lzry', 
                              threshold=0.01, colorbar=True, plot_abs=False)
    plt.savefig(output_png)
    plt.close()

# Process each participant's data
def process_subject(subject_id, output_folder, mkrr_pipeline):
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    output_sub_dir = os.path.join(output_folder, subject_id)
    
    # **Skip processing if subject's output folder already exists with results**
    if os.path.exists(output_sub_dir) and any(f.endswith('.nii.gz') for f in os.listdir(output_sub_dir)):
        print(f"Skipping {subject_id}: Results already exist.")
        return
    
    os.makedirs(output_sub_dir, exist_ok=True)
    
    Y_train_path = f'movieDM_subdata/sub-{subject_id}_movieDM_wholebrain.hdf'
    Y_test_path = f'movieTP_subdata/sub-{subject_id}_movieTP_wholebrain.hdf'

    with h5py.File(Y_train_path, "r") as f:
        Y_train = f["fmri_response"][:]
    Y_train = Y_train.astype("float32")
    print("Y_train", Y_train.shape)

    with h5py.File(Y_test_path, "r") as f:
        Y_test = f["fmri_response"][:]
    Y_test = Y_test.astype("float32")
    print("Y_test", Y_test.shape)
    
    Y_train -= Y_train.mean(0)
    Y_test -= Y_test.mean(0)
    
    feature_split_rscores_z = {name: [] for name in CONS}

    print(f"\nProcessing MKRR_pipeline for Subject: {subject_id}")
    
    #print("Training data", X_train.shape, Y_train.shape)  #Training data (750, 10) (750, 146585)
    #print("Testing data", X_test.shape, Y_test.shape) #Testing data (250, 10) (250, 146585)
    
    # model fitting & prediction
    mkrr_pipeline.fit(X_train, Y_train)

    Y_pred = mkrr_pipeline.predict(X_test) 
    r_scores = backend.to_numpy(correlation_score(Y_test,Y_pred))
    print("r_scores", r_scores.shape)
    
    # fisher-z transformation
    r_scores_z = np.arctanh(r_scores)
    
    save_nifti_and_plot(r_scores_z, f"{subject_id}_whole_model_zscore", output_sub_dir)
    
    Y_pred_split = mkrr_pipeline.predict(X_test, split=True) 
    r_scores_split = backend.to_numpy(correlation_score_split(Y_test, Y_pred_split))
    print("r_scores_split", r_scores_split.shape)
    
    # Save fisher-z transformed results
    for i, feature_name in enumerate(CONS):
        feature_split_rscores_z[feature_name].append(np.arctanh(r_scores_split[i]))
    
    for feature_name, z_scores in feature_split_rscores_z.items():
        save_nifti_and_plot(z_scores, f"{subject_id}_{feature_name}_zscore", output_sub_dir)
    
    return subject_id

# Main execution
def main():
    
    output_folder = "movieTP_lexisurp_10feature_crossresult/" 
    os.makedirs(output_folder, exist_ok=True)
    
    mkrr_pipeline = create_pipeline()
    
    subject_ids = pd.read_csv('movieTP_subinfo.csv')['ID'].astype(str).tolist()
    print("total number of subjects:",len(subject_ids))

    for sub in subject_ids:
        process_subject(sub, output_folder, mkrr_pipeline)
        
    print("Processing complete!")

if __name__ == "__main__":
    main()
