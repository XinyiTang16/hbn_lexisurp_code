# ------------------------------------------------------------
# This script estimates developmental trajectories of lexical surprisal encoding across functional networks using generalized additive models (GAM).
#
# Analysis pipeline:
# 1. Fit a GAM in which surprisal encoding is predicted by age, with network-specific smooth functions of age:
#       encoding ~ network + s(age, by = network) + covariates
#
# 2. Generate predicted developmental trajectories and 95% CI for each network while holding other covariates at reference values.
#
# 3. Plot network-specific growth curves together with raw data for each network.
#
# 4. Compute derivatives of the age smooth to estimate the developmental rate of change and identify age ranges where the slope significantly differs from zero.
#
# 5. Test whether developmental trajectories differ across networks by comparing:
#       - Full model: network-specific age smooths
#       - Reduced model: shared age smooth across networks
#    using an ANOVA deviance test.
#
# 6. Perform bootstrap simulations under the reduced model to obtain an empirical p-value for the deviance difference.
#
# Input data format:
# The input CSV should be a LONG-format dataframe where each row represents one subject–network observation. 
# Each subject therefore appears in multiple rows (one per network). 
# The file should include:
#   - subject information (e.g., ID, age, sex, site, handedness, motion)
#   - encoding value for that network
#   - network label (e.g., Language, MDN, ToM)
#
# Example structure of the input data file:
#   ID | Age | Sex | Site | FD | encoding | network
#   -----------------------------------------------
#   S01 | ... | ... | ... | ... | 0.12 | Language
#   S01 | ... | ... | ... | ... | 0.05 | MDN
#   S01 | ... | ... | ... | ... | 0.04 | ToM
#
# This script uses Despicable Me (DM) as an example dataset.
# The same analysis pipeline is used for The Present (TP) by simply changing the input data file.
# ------------------------------------------------------------

# required packages
library(mgcv)
library(gratia)
library(MASS)      
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)  
library(gratia)
library(parallel)
library(scales)

set.seed(123)   # for reproducibility
# -------------------------
# pathway
datafile <- "movieDM_subinfo_and_TOP10percROI_n1415reshaped_encoding.csv" #the input long dataframe file
datafolder <- "~/3_movieDM_GAM/"
setwd(datafolder)
destfolder <- file.path(datafolder, "GAM_combined_ANOVA_and_bootstrap_plot")
dir.create(destfolder, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# 0. Read & prepare
data <- read.csv(datafile)

# filtering: drop rows with NA in any column that will use
needed_cols <- c("encoding", "network", "MRI_Track.Age_at_Scan",
                 "Basic_Demos.Sex", "MRI_Track.Scan_Location",
                 "EHQ.EHQ_Total", "FD_mean_DM", "ID")
filtered_data <- data[complete.cases(data[, intersect(names(data), needed_cols)]), ]

filtered_data$Basic_Demos.Sex <- as.factor(filtered_data$Basic_Demos.Sex)
filtered_data$MRI_Track.Scan_Location <- as.factor(filtered_data$MRI_Track.Scan_Location)
filtered_data$network <- as.factor(filtered_data$network)

# Quick check of levels
levels(filtered_data$network)

# -------------------------
# 1. Fit the FULL combined GAM (separate smooths by network)

combinedmod_gam <- gam(encoding ~ network +
                         s(MRI_Track.Age_at_Scan, bs = "cs", by = network) +
                         Basic_Demos.Sex + MRI_Track.Scan_Location + EHQ.EHQ_Total + FD_mean_DM,
                       data = filtered_data,
                       method = "REML",
                       na.action = na.omit)

#get k info and model summary
kinfo <- capture.output(gam.check(combinedmod_gam, rep = 200))
summary_model <- capture.output(summary(combinedmod_gam))

#stored formatted summary
all_summaries <- c(paste0("========== Model Summary =========="), "k check: ",kinfo, "Model summary: ", summary_model, "")

writeLines(all_summaries, con = paste0(destfolder,"/movieDM_GAM_summaries.txt", sep="" )) 

# -------------------------
# 2. Prediction grid (same covariate references for all networks)
age_grid <- seq(min(filtered_data$MRI_Track.Age_at_Scan, na.rm = TRUE),
                max(filtered_data$MRI_Track.Age_at_Scan, na.rm = TRUE),
                length.out = 1000)

# choose reference values for other covariates
ref_sex <- levels(filtered_data$Basic_Demos.Sex)[1]
ref_site <- levels(filtered_data$MRI_Track.Scan_Location)[1]
ref_ehq <- mean(filtered_data$EHQ.EHQ_Total, na.rm = TRUE)
ref_fd  <- mean(filtered_data$FD_mean_DM, na.rm = TRUE)

net_levels <- levels(filtered_data$network)

# build newdata for each network
newdata_all <- do.call(rbind, lapply(net_levels, function(net) {
  data.frame(
    MRI_Track.Age_at_Scan = age_grid,
    network = factor(net, levels = net_levels),
    Basic_Demos.Sex = factor(ref_sex, levels = levels(filtered_data$Basic_Demos.Sex)),
    MRI_Track.Scan_Location = factor(ref_site, levels = levels(filtered_data$MRI_Track.Scan_Location)),
    EHQ.EHQ_Total = ref_ehq,
    FD_mean_DM = ref_fd,
    network_label = net
  )
}))

# -------------------------
# 3. Get predictions and standard errors using original method
# Generates predicted values from GAM model for each row of G1_pred
G1_pred <-cbind(newdata_all,
                predict(combinedmod_gam,
                        newdata=newdata_all,
                        se.fit = TRUE,
                        type = "link",
                        exclude = c("Basic_Demos.Sex", "MRI_Track.Scan_Location", "FD_mean_DM",
                                    "EHQ.EHQ_Total")))
G1_pred$conf.low <- G1_pred$fit - G1_pred$se.fit*1.96
G1_pred$conf.high <- G1_pred$fit + G1_pred$se.fit*1.96

# -------------------------
# 4. Plot the fitted smooths for each network
# custom colors
color_map <- c(
  "Language" = "#dc3f40",
  "MDN"      = "#0d53a2",
  "ToM"      = "#0F6324"
)

# one plot per network, with raw points + smooth + CI
for (net in net_levels) {
  
  df_points <- filtered_data %>% dplyr::filter(network == net)
  df_pred <- G1_pred %>% dplyr::filter(network_label == net)
  
  p <- ggplot() +
    geom_point(data = df_points, aes(x = MRI_Track.Age_at_Scan, y = encoding), color = color_map[[net]],alpha = 0.30, size = 2.7) +
    geom_line(data = df_pred, aes(x = MRI_Track.Age_at_Scan, y = fit), color = color_map[[net]], linewidth = 3.5) +
    geom_ribbon(data = df_pred, aes(x = MRI_Track.Age_at_Scan, ymin = conf.low, ymax = conf.high),fill = color_map[[net]],alpha = 0.4) +
    labs(x = "Age (years)", y = "Surprisal encoding (z-transformed)") +
    coord_cartesian(ylim = c(0, 0.20), xlim = c(5, 22)) +
    scale_x_continuous(breaks = seq(5, 22, by = 2)) +
    scale_y_continuous(breaks = seq(0, 0.20, by = 0.05)) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      legend.position = "none",
      axis.text = element_text(size = 23),
      axis.line = element_line(color = "black", linewidth = 2),
      axis.ticks = element_line(color = "black", linewidth = 1),
      axis.ticks.length.x = unit(5, "pt"),
      axis.ticks.length.y = unit(-5, "pt"),
      axis.title.x = element_text(size = 26, margin = margin(t = 12)),
      axis.title.y = element_text(size = 26, margin = margin(r = 18))
    )
  
  ggsave(
    filename = file.path(destfolder, paste0("Fig_GAM_", net, "_points_plus_smooth.png")),
    plot = p,
    width = 10,
    height = 7,
    dpi = 300
  )
}

# -------------------------
# 5. Save residuals from the full combined model
filtered_data$resid <- residuals(combinedmod_gam, type = "response")
write.csv(filtered_data, file = paste0(destfolder, "/movieDM_GAM_Network_addresiduals.csv"), row.names = FALSE)

# -------------------------
# 6. get derivatives 
# Step 1: Compute the derivative of the age smooth
deriv_list <- lapply(net_levels, function(net) {
  d <- derivatives(combinedmod_gam,
                   select = paste0("s(MRI_Track.Age_at_Scan):network", net),
                   interval = "simultaneous", #simultaneous CI
                   level = 0.95,
                   n = 1000,
                   n_sim = 10000)  #simulation 10000
  d$network <- net
  return(d)
})
deriv_all <- do.call(rbind, deriv_list)

# Step 2: Mark where the derivative is significantly non-zero (i.e., slope is significantly ≠ 0)
deriv_all <- deriv_all %>%
  mutate(sig = !(0 > .lower_ci & 0 < .upper_ci))  # TRUE if 95% CI excludes 0

#add: save
write.csv(deriv_all, file = sprintf("%s/movieDM_GAM_derivatives.csv", destfolder), row.names = FALSE)

# Step 3: Report significant age ranges (per network)
report_df <- deriv_all %>%
  group_by(network) %>%
  summarize(
    any_sig   = any(sig),
    lower_age = if (any(sig)) min(.data[['MRI_Track.Age_at_Scan']][sig]) else NA_real_,
    upper_age = if (any(sig)) max(.data[['MRI_Track.Age_at_Scan']][sig]) else NA_real_,
    .groups = "drop"
  )

# print concise messages
lines <- apply(report_df, 1, function(r) {
  if (isTRUE(as.logical(r[["any_sig"]]))) {
    sprintf("Network %s: Significant slope (derivative ≠ 0) somewhere between %.2f and %.2f years.",
            r[["network"]], as.numeric(r[["lower_age"]]), as.numeric(r[["upper_age"]]))
  } else {
    sprintf("Network %s: No significant non-zero derivative detected.", r[["network"]])
  }
})

outfile <- file.path(destfolder, "movieDM_network_derivative_significance_report.txt")
writeLines(lines, outfile)

# Step 4: Plot derivatives (one for each network)
# derivative plot per network, with significance stripes (sig==TRUE) + alpha scaled by |derivative|
y_stripe_bottom <- -0.006
y_stripe_top    <- -0.005
# one derivative plot per network
for (net in net_levels) {
  
  df_deriv <- deriv_all %>% dplyr::filter(network == net)
  
  # build stripe df (only significant ages). If none significant, this will be empty -> no stripes drawn.
  stripe_df_deriv <- df_deriv %>% dplyr::filter(sig) %>% dplyr::transmute(age = MRI_Track.Age_at_Scan, slope_mag = scales::rescale(abs(.derivative), to = c(0, 1)))
  
  p_deriv <- ggplot(df_deriv,aes(x = MRI_Track.Age_at_Scan, y = .derivative)) +
    geom_line(color = color_map[[net]], linewidth = 3.5) +
    geom_ribbon(aes(ymin = .lower_ci, ymax = .upper_ci), fill = color_map[[net]], alpha = 0.4, color = NA) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1.5, color = "black") +
    geom_segment(data = stripe_df_deriv,aes(x = age, xend = age, y = y_stripe_bottom, yend = y_stripe_top,alpha = slope_mag),
              color = color_map[[net]],linewidth = 0.8,inherit.aes = FALSE) +
    #scale_alpha_continuous(range = c(0.15, 0.95), guide = "none")
    labs(x = "Age (years)", y = "Developmental rate") +
    coord_cartesian(ylim = c(-0.006, 0.010), xlim = c(5, 22)) +
    scale_x_continuous(breaks = seq(5, 22, by = 2)) +
    scale_y_continuous(breaks = seq(-0.006, 0.010, by = 0.002)) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      legend.position = "none",
      axis.text = element_text(size = 20),
      axis.line = element_line(color = "black", linewidth = 2),
      axis.ticks.x = element_line(color = "black", linewidth = 1),
      axis.ticks.y = element_line(color = "black", linewidth = 1),
      axis.ticks.length.x = unit(5, "pt"),
      axis.ticks.length.y = unit(-5, "pt"),
      axis.title.x = element_text(size = 26, margin = margin(t = 12)),
      axis.title.y = element_text(size = 26, margin = margin(r = 18))
    )
  
  ggsave(
    file.path(destfolder, paste0("Fig_GAM_derivative_", net, ".png")),
    plot = p_deriv, width = 10, height = 7, dpi = 300
  )
}

# -------------------------
# 7. Build Reduced GAM model & perform ANOVA & Bootstrap on deviance 
combinedmod_gam_reduced <- gam(encoding ~ network +
                                 s(MRI_Track.Age_at_Scan, bs = "cs") +
                                 Basic_Demos.Sex + MRI_Track.Scan_Location + EHQ.EHQ_Total + FD_mean_DM,
                               data = filtered_data,
                               method = "REML",
                               na.action = na.omit)

## Observed deviance increase from anova.gam (reduced vs. full)
res <- anova.gam(combinedmod_gam_reduced, combinedmod_gam, test = "Chisq")

outfile <- file.path(destfolder, "movieDM_ANOVA_gam_result.txt")
writeLines(capture.output(res), outfile)
# observed value
obs_df  <- as.data.frame(res)
obs_stat <- as.numeric(obs_df$Deviance[2])

# -------------------------
# 8. Perform Bootstrapping on Deviance difference

# Bootstrapping setting
B <- 1000       # number of bootstrap samples
cores <- max(1, detectCores() - 1)

# --- bootstrap loop ---
boot_stats <- numeric(B)

boot_one <- function(i) {
  # simulate response under reduced model
  # null hypothesis: reduced model == full model
  sim_y <- simulate(combinedmod_gam_reduced, nsim = 1)[,1]
  
  # rebuild data with simulated response
  sim_data <- filtered_data
  sim_data$encoding <- sim_y
  
  # refit reduced and full models
  sim_reduced <- try(gam(encoding ~ network +
                           s(MRI_Track.Age_at_Scan, bs="cs") +
                           Basic_Demos.Sex + MRI_Track.Scan_Location +
                           EHQ.EHQ_Total + FD_mean_DM,
                         data = sim_data, method="REML", na.action=na.omit),
                     silent = TRUE)
  if (inherits(sim_reduced, "try-error")) return(NA_real_)
  
  sim_full <- try(gam(encoding ~ network +
                        s(MRI_Track.Age_at_Scan, by=network, bs="cs") +
                        Basic_Demos.Sex + MRI_Track.Scan_Location +
                        EHQ.EHQ_Total + FD_mean_DM,
                      data = sim_data, method="REML", na.action=na.omit),
                  silent = TRUE)
  if (inherits(sim_full, "try-error")) return(NA_real_)
  
  # ANOVA deviance drop for this simulated dataset
  atab <- try(anova.gam(sim_reduced, sim_full, test = "Chisq"), silent = TRUE)
  if (inherits(atab, "try-error")) return(NA_real_)
  adf <- as.data.frame(atab)
  
  as.numeric(adf$Deviance[2])  # deviance drop (Model 2 row)
}

boot_vec <- unlist(mclapply(seq_len(B), boot_one, mc.cores = cores))
boot_stats <- boot_vec[is.finite(boot_vec)]

# --- p-value ---
p_boot <- (sum(boot_stats >= obs_stat) + 1) / (length(boot_stats) + 1)

cat("Observed test statistic =", obs_stat, "\n")
cat("Bootstrap p-value =", p_boot, "\n")

# summary
summary_df <- data.frame(
  observed_stat = as.numeric(obs_stat),
  bootstrap_p   = p_boot,
  n_boot_used   = length(boot_stats)
)

write.csv(summary_df,
          file = file.path(destfolder, "movieDM_bootstrap_DeviDiff_summary.csv"),
          row.names = FALSE)

# full bootstrap distribution
write.csv(data.frame(boot_stat = boot_stats),
          file = file.path(destfolder, "movieDM_bootstrap_DeviDiff_distribution.csv"),
          row.names = FALSE)

#draw plot
# data frames
df_all  <- data.frame(boot_stat = boot_stats)
df_tail <- subset(df_all, boot_stat >= as.numeric(obs_stat))

# choose a sensible binwidth (Freedman–Diaconis), with fallback
bw <- tryCatch(
  2 * IQR(df_all$boot_stat) / length(df_all$boot_stat)^(1/3),
  error = function(e) NA_real_
)
if (!is.finite(bw) || bw <= 0) bw <- diff(range(df_all$boot_stat)) / 30

# build the plot
p_hist <- ggplot(df_all, aes(x = boot_stat)) +
  geom_histogram(binwidth = bw, fill = "grey80", color = "grey40") +
  geom_histogram(data = df_tail, binwidth = bw, fill = "tomato", color = "grey40") +
  geom_vline(xintercept = as.numeric(obs_stat), linetype = "dashed") +
  labs(
    title = "Bootstrap null distribution of Deviance Diff",
    subtitle = sprintf("Observed ΔDeviance = %.3f, bootstrap p = %.4f (tail shaded)", obs_stat, p_boot),
    x = "Difference in deviance (reduced - full)",
    y = "Count"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(destfolder, "movieDM_bootstrap_deviance_hist.png"),
       plot = p_hist, width = 7, height = 4.5, dpi = 300)

