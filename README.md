# hbn_lexisurp_code

This repository contains the analysis pipeline for manuscript: "Contextual predictability in development: Increasing language-network encoding during naturalistic speech comprehension", using fMRI and behavioral data from the Healthy Brain Network dataset.

## Overview

The pipeline is organized into seven analysis stages:

1. **0_video_feature**: Extract and preprocess movie feature regressors.
2. **1_encoding_model**: Fit voxelwise encoding models using 10 movie features.
3. **2_lexisurp_selectivity**: Perform whole-brain and ROI-level lexical surprisal encoding analyses.
4. **3_GAM**: Estimate developmental trajectories of surprisal encoding using generalized additive models (GAMs).
5. **4_behavioral_prediction**: Test whether network-level surprisal encoding predicts behavioral measures.
6. **5_IS_RSA**: Conduct inter-subject representational similarity analysis (IS-RSA) on language-network neural similarity matrices and hypothesis-based developmental model matrices.
7. **6_ISPS_segment**: Conduct inter-subject phase synchrony (ISPS) analyses and test whether age-related synchrony segments are associated with surprisal features.

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

### `0_video_feature`

Builds the movie feature regressors used in the encoding models.
- `1_get_linguistic_regressor.ipynb`: extracts linguistic features (lexical surprisal and word frequency) from the movie stimulus.
- `2_process_features.ipynb`: resamples and combines the regressors used in the encoding model

### `1_encoding_model`

Fits voxelwise multiple-kernel ridge regression (MKRR) encoding models with delayed feature kernels.
-  `regressors`: 10 regressors for each movie used in the encoding model
- `1_movieDM_mkrr_pipe_5foldCV.py`: 5-fold cross-validated encoding analysis within DM.
- `2_movieTP_mkrr_pipe_crossCV.py`: cross-movie encoding model validation analysis with weights trained on DM and tested on TP.

### `2_lexisurp_selectivity`

Summarizes lexical surprisal encoding results at the whole-brain and ROI level.
- `unthres_ROI/`: unthresholded ROI files for language, multiple demand (MD), and theory of mind (ToM) networks.
- `wholebrain_and_ROI_analysis.ipynb`: computes group-level whole-brain maps, as well as ROI-level results from subject-level encoding outputs.

### `3_GAM`

Characterizes developmental trajectories of network-level lexical surprisal encoding.
- `GAM_analysis_with_plots.R`: fits GAMs, estimates derivatives, and examines age-by-network interaction significance.

### `4_behavioral_prediction`

Tests whether age-residualized network-level surprisal encoding predicts behavioral measures.
- `behavior_predict.py`: runs bootstrapped prediction and permutation testing for language and nonverbal IQ measures using network-level encoding estimates.

### `5_IS_RSA`

Performs inter-subject representational similarity analysis (IS-RSA).
- `1_extract_timecourse_ssROI_DM.py`: extracts subject-specific ROI mean time series and phase-angle time series for DM using the top 10% of voxels within each ROI based on lexical surprisal encoding maps.
- `1_extract_timecourse_ssROI_TP.py`: extracts the corresponding TP time series using ROIs defined from DM lexical surprisal maps.
- `2_ISRSA.ipynb`: builds developmental similarity models and performs IS-RSA.

### `6_ISPS_segment`

Performs inter-subject phase synchrony (ISPS) analyses.
- `ISPS_analysis.ipynb`: computes age-binned ISPS from ROI phase-angle time series, identifies time segments in which ISPS changes with age, and tests whether those age-related segments are associated with lexical surprisal features.

## Data

The raw imaging, demographic, and behavioral data used in this analysis are not included in this repository. Information about accessing the Healthy Brain Network (HBN) dataset is available at: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/.

## Notes

- Most analyses are written with DM as the primary example dataset.
- Parallel TP analyses generally reuse the same code with different inputs.
- Some scripts assume local helper files such as `delayer.py` and interpolation utilities are available in the working environment.