# hbn_lexisurp_code

This repository contains the analysis pipeline for manuscript: "Contextual predictability in development: Increasing language-network encoding during naturalistic speech comprehension", using fMRI and behavioral data from the Healthy Brain Network (HBN) dataset.

## Overview

The pipeline is organized into seven analysis stages:

1. **0_video_feature**: Extract and preprocess movie feature regressors.
2. **1_encoding_model**: Fit voxelwise encoding models using 10 movie features.
3. **2_lexisurp_selectivity**: Analyze lexical surprisal encoding at whole-brain and ROI levels.
4. **3_GAM**: Estimate developmental trajectories of surprisal encoding using generalized additive models (GAMs).
5. **4_behavioral_prediction**: Test whether network-level surprisal encoding predicts behavioral measures.
6. **5_IS_RSA**: Perform inter-subject representational similarity analysis (IS-RSA) between neural and developmental model matrices
7. **6_ISPS_segment**: Analyze inter-subject phase synchrony (ISPS) and its association with surprisal feature.

## Environment

The scripts in this repository were run with Python 3.12.2 and RStudio 2024.04.2+764 on a Mac OS arm64 system.

The Python environment file is provided in `environment.yml`. It contains the package versions used in the Python scripts and notebooks in this repository.

To create the Python environment:

```bash
conda env create -f environment.yml
conda activate hbn_lexisurp_code
```
The R packages used for the GAM analysis are listed at the beginning of the script. You can install them using the `install.packages()` function in R.

## Folder structure

#### `0_video_feature`
- `1_get_linguistic_regressor.ipynb`: extracts linguistic features from the movie stimulus.
- `2_process_features.ipynb`: resamples and combines the regressors for the encoding model

#### `1_encoding_model`
-  `regressors/`: 10 regressors for each movie used in the encoding model
- `1_movieDM_mkrr_pipe_5foldCV.py`: 5-fold cross-validated encoding analysis within DM.
- `2_movieTP_mkrr_pipe_crossCV.py`: cross-movie encoding model validation (trained on DM and tested on TP).

#### `2_lexisurp_selectivity`
- `unthres_ROI/`: unthresholded ROI masks for language, multiple demand (MD), and theory of mind (ToM) networks.
- `wholebrain_and_ROI_analysis.ipynb`: computes group-level whole-brain and ROI-level results from subject-level encoding outputs.

#### `3_GAM`
- `GAM_analysis_with_plots.R`: fits GAMs, estimates derivatives, and tests age-by-network interactions.

#### `4_behavioral_prediction`
- `behavior_predict.py`: performs bootstrapped prediction and permutation testing for language and nonverbal IQ.

#### `5_IS_RSA`
- `1_extract_timecourse_ssROI_DM.py`: extracts subject-specific ROI mean and phase-angle time series from DM 
- `1_extract_timecourse_ssROI_TP.py`: extracts TP time series using ROIs defined from DM surprisal maps.
- `2_ISRSA.ipynb`: builds developmental similarity models and runs IS-RSA.

#### `6_ISPS_segment`
- `ISPS_analysis.ipynb`: computes age-binned ISPS and tests whether age-related synchrony segments are associated with lexical surprisal features.

## Data

In accordance with the HBN Data Usage Agreement, the original imaging and behavioral data used in this study cannot be redistributed without authorized access. Researchers interested in accessing the dataset can find information about obtaining the HBN dataset at: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/.