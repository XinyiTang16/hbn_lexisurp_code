"""
Behavioral prediction from network-level surprisal encoding

This script tests whether age-residualized network-level lexical surprisal encoding predicts behavioral measures (language ability and nonverbal IQ) 
using a bootstrapped prediction framework with permutation testing.

Input data:
The input CSV should be a LONG-format dataframe where each row correspondto a subject-network observation and includes:
    - network label (e.g., Language, MDN, ToM)
    - surprisal encoding residual ("resid")
    - behavioral scores (CELF.CELF_Total, WISC.WISC_MR_Scaled)

Each subject therefore appears multiple times in the dataset (one row per network).

Note:
This script uses Despicable Me (DM) as an example dataset. 
The same analysis pipeline is applied to The Present (TP) by replacing the input data file.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Config
# ==============================
seed = 42
np.random.seed(seed)

DATA_CSV = "~/movieDM_GAM_Network_addresiduals.csv"
OUTROOT  = "~/4_movieDM_behav_prediction/"

Y_LIST   = ["CELF.CELF_Total", "WISC.WISC_MR_Scaled"]
AGE_COL  = "MRI_Track.Age_at_Scan"
CELF_COL = "CELF.CELF_Total"

N_BOOT   = 1000
N_PERM   = 10000
TEST_SIZE = 0.5
RANDOM_STATE_BASE = 0  # seeds are 0..N_BOOT-1 (and 0..N_PERM-1 for perms)

# ==============================
# Helpers
# ==============================
def ci95(x, axis=0):
    return np.percentile(x, [2.5, 97.5], axis=axis)

def stars_from_p(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def safe_pearsonr(a, b):
    # handle constant vectors gracefully
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    r, _ = pearsonr(a, b)
    return r

os.makedirs(OUTROOT, exist_ok=True)

# ==============================
# Load data
# ==============================
df = pd.read_csv(DATA_CSV)

networks = df['network'].dropna().unique()
print("Networks in the data:", networks)

palette = {
    "CELF.CELF_Total": "#ffa43bd7",   # orange
    "WISC.WISC_MR_Scaled": "#838180e9"   # gray
}

summary_rows = []

# ==============================
# Main loop
# ==============================
for net in networks:
    print(f"\n=== Processing network: {net} ===")
    net_dir = os.path.join(OUTROOT, f"{net}_only_Prediction_fisherz")
    os.makedirs(net_dir, exist_ok=True)

    # For the combined r-value boxplot across targets (within this network)
    r_boot_dict = {}     # target -> array of r over boot
    p_perm_dict = {}     # target -> p value
    mean_r_dict = {}     # target -> mean r over boot

    # coefficient summaries across targets
    coef_summary_rows = []

    for y_name in Y_LIST:
        
        print(f"\nTarget: {y_name}")
        used = df.loc[df["network"] == net, [y_name, AGE_COL, "resid"]].dropna()
        print(f"Total N subjects: {len(used)}")
        
        # predictor stays the same
        X = used[["resid"]].values #predictor: residual from GAM for this network
        
        # change Y for CELF_Total
        if y_name == "CELF.CELF_Total":
            
            # if CELF: regress CELF_Total on Age (within this network)
            age_model = LinearRegression()
            age_model.fit(used[[AGE_COL]].values, used[y_name].values)

            # residualized behavioral outcome
            y_resid = used[y_name].values - age_model.predict(used[[AGE_COL]].values)

            y = y_resid
        else:
            # if nonverbal IQ, keep scaled score
            y = used[y_name].values

        # ---------- Bootstrapped 50% holdout ----------
        r_values = []
        coefs = []  # store slope(s)

        for seed in range(N_BOOT):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=seed
            )
            model = LinearRegression()
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            r = safe_pearsonr(y_pred, y_te)
            
            # add fisher z transform
            r_values.append(np.arctanh(r))
            coefs.append(model.coef_.ravel())  

        r_values = np.array(r_values)             
        coefs = np.array(coefs).reshape(N_BOOT, -1)  

        # coefficient summary (mean & 95% CI)
        coef_mean = coefs.mean(axis=0)            
        coef_ci   = ci95(coefs, axis=0)           
        predictor_names = [f"resid_{net}"]

        for i, pname in enumerate(predictor_names):
            print(f"{pname} coef: mean={coef_mean[i]:.4f}, 95% CI=({coef_ci[0,i]:.4f}, {coef_ci[1,i]:.4f})")
            coef_summary_rows.append({
                "network": net,
                "target": y_name,
                "predictor": pname,
                "coef_mean": coef_mean[i],
                "coef_ci_low": coef_ci[0, i],
                "coef_ci_high": coef_ci[1, i]
            })

        # Save per-target coefficients over boot
        pd.DataFrame(coefs, columns=predictor_names).to_csv(
            os.path.join(net_dir, f"{y_name}_coefficients_boot{N_BOOT}.csv"), index=False
        )

        # ---------- Permutation test (shuffle y) ----------
        r_null = []
        for i in tqdm(range(N_PERM), desc=f"Perm test ({y_name})"):
            y_perm = np.random.permutation(y)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_perm, test_size=TEST_SIZE, random_state=i
            )
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            r = safe_pearsonr(model.predict(X_te), y_te)
            r_null.append(np.arctanh(r))  # fisher-z

        r_null = np.array(r_null)
        # Back-transform mean r_values from fisher z
        real_mean_r = float(np.tanh(np.mean(r_values)))
        # One-sided: how often null >= observed mean r
        p_perm = float(np.mean(np.tanh(r_null) >= real_mean_r))

        print(f"Mean true r = {real_mean_r:.4f}")
        print(f"Permutation p-value = {p_perm:.8g}")

        # Save null distribution plot
        plt.figure(figsize=(10,7))
        plt.hist(r_null, bins=50, alpha=0.7, label="Null")
        plt.axvline(real_mean_r, linestyle='--', color='red', label=f"Mean true r = {real_mean_r:.3f}")
        plt.xlabel("r")
        plt.ylabel("Frequency")
        plt.title(f"Permutation Test: {net} → {y_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(net_dir, f'{y_name}_perm{N_PERM}_hist.png'), dpi=200)
        plt.close()

        # Save the arrays
        np.save(os.path.join(net_dir, f'{y_name}_r_boot{N_BOOT}.npy'), r_values)
        np.save(os.path.join(net_dir, f'{y_name}_rnull_perm{N_PERM}.npy'), r_null)

        # save boot r's
        pd.DataFrame({"r": r_values}).to_csv(
            os.path.join(net_dir, f'{y_name}_r_boot{N_BOOT}.csv'), index=False
        )

        # Track for summary and for the combined boxplot
        r_boot_dict[y_name] = r_values
        p_perm_dict[y_name] = p_perm
        mean_r_dict[y_name] = real_mean_r

        # Add a summary row for this network × behavior
        summary_rows.append({
            "network": net,
            "target": y_name,
            "mean_true_r": real_mean_r,
            "perm_p": p_perm,
            "n_boot": N_BOOT,
            "n_perm": N_PERM,
            "n_subjects": len(used)
        })

    # ---------- Save per-network coefficient summary ----------
    if coef_summary_rows:
        pd.DataFrame(coef_summary_rows).to_csv(
            os.path.join(net_dir, f'coeff_summary_boot{N_BOOT}.csv'), index=False
        )

    # ---------- Combined boxplot of bootstrap r for this network, with sig marks ----------
    if len(r_boot_dict) > 0:
        # long format for seaborn
        df_long = pd.concat([
            pd.DataFrame({"Target": t, "r": r_boot_dict[t]})
            for t in r_boot_dict
        ], ignore_index=True)
        
        plt.figure(figsize=(12, 7))
        # Build y_data 
        order = ["CELF.CELF_Total", "WISC.WISC_MR_Scaled"]
        xlabels = ["Language", "Nonverbal IQ"]

        y_data = [
            df_long.loc[df_long["Target"] == order[0], "r"].dropna().to_numpy(),
            df_long.loc[df_long["Target"] == order[1], "r"].dropna().to_numpy()
        ]

        POSITIONS = np.arange(len(order))
        COLORS = ["#ffba6bee", "#a3a3a3e9"]  # orange, gray

        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Violin plot
        violins = ax.violinplot(
            y_data,
            positions=POSITIONS,
            widths=0.6,
            bw_method="silverman",
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for pc, color in zip(violins["bodies"], COLORS):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_linewidth(0)
            pc.set_alpha(0.8)

        # Thin box overlay
        medianprops = dict(linewidth=3.0, color="black")
        boxprops    = dict(linewidth=2.0, color="black")

        ax.boxplot(
            y_data,
            positions=POSITIONS,
            widths=0.15,
            showfliers=False,
            showcaps=False,
            medianprops=medianprops,
            whiskerprops=boxprops,
            boxprops=boxprops
        )

        # Labels / styling
        ax.set_ylim([-0.15, 0.30])
        ax.set_xticks(POSITIONS)
        ax.set_xticklabels(xlabels, fontsize=23)
        ax.set_ylabel("Split-half prediction correlation", fontsize=23)
        ax.set_xlabel("")

        ax.tick_params(direction="in", axis="y", labelsize=16, width=2)
        ax.tick_params(direction="out", axis="x", length=10, width=2, pad=12)
        ax.spines["bottom"].set_linewidth(4)
        ax.spines["left"].set_linewidth(4)

        sns.despine(top=True, right=True)
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(net_dir, f'bootstrap_r_boxplot_net-{net}_boot{N_BOOT}.png'), dpi=300)
        plt.close()

# ==============================
# Global summary (all networks × targets)
# ==============================
summary_df = pd.DataFrame(summary_rows).sort_values(["network", "target"])
summary_csv = os.path.join(OUTROOT, f'prediction_perm_summary_allnets_boot{N_BOOT}_perm{N_PERM}.csv')
summary_df.to_csv(summary_csv, index=False)
print("\nSaved global summary to:", summary_csv)