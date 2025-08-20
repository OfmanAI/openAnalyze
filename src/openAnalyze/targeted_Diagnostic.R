#!/usr/bin/env Rscript
# targeted_Diagnostic.R
# -----------------------------------------------------------------------------
# Balanced evaluation for DeepMedia detector scores.
# Now supports user-defined thresholds and fakes-only analysis.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(pROC)
  library(ROCR)
  library(ggplot2)
})

# ── 1. CLI -------------------------------------------------------------------
opt <- OptionParser(option_list = list(
  make_option("--single", help="Path to the combined CSV data."),
  make_option("--out", default = ".", help="Directory to save output files."),
  make_option("--use-threshold", type = "double", default = NULL, help = "Specify a threshold to use instead of F1 optimization."),
  make_option("--n_iter",  type = "integer", default = 200),
  make_option("--step",    type = "double",  default = 0.01),
  make_option("--suffix", default = "", help = "Suffix to append to output filenames before the extension.")
))
args <- parse_args(opt)
set.seed(42)

out_dir    <- normalizePath(args$out, mustWork = FALSE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
setwd(out_dir)

log <- function(msg) message(sprintf("[%s] %s", format(Sys.time(), "%H:%M:%S"), msg))

# --- THIS IS THE FIX ---
# Create a suffix string (e.g., "_results_v1_video_full") to append to filenames.
file_suffix <- if (nchar(args$suffix) > 0) paste0("_", args$suffix) else ""

# ── 2. Load data -------------------------------------------------------------
log("Loading detector CSV")
if (is.null(args$single)) stop("A --single CSV must be provided.")

df <- read.csv(args$single)
log(paste("Loaded", nrow(df), "rows from", basename(args$single)))
if (!"score" %in% names(df)) stop("Input CSV needs a 'score' column.")
if (!"true_label" %in% names(df)) stop("Input CSV must contain a 'true_label' column (1=fake, 0=real).")


# ── 3. Helper functions & Initial Setup --------------------------------------
evaluate_once_f1 <- function(df_in, step = 0.01) {
  best_f1 <- -Inf; best_thr <- NA
  for (t in seq(0, 1, by = step)) {
    prd <- df_in$score >= t
    tp  <- sum(prd & df_in$true_label == 1)
    fp  <- sum(prd & df_in$true_label == 0)
    fn  <- sum(!prd & df_in$true_label == 1)
    prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    rec  <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1   <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
    if (!is.na(f1) && f1 > best_f1) {
      best_f1 <- f1
      best_thr <- t
    }
  }
  return(best_thr)
}

find_best_accuracy_threshold <- function(df_in, step = 0.01) {
  best_acc <- -Inf; best_thr <- NA;
  n_total <- nrow(df_in)
  for (t in seq(0, 1, by = step)) {
    prd <- df_in$score >= t
    tp  <- sum(prd &  df_in$true_label == 1)
    tn  <- sum(!prd & df_in$true_label == 0)
    acc <- (tp + tn) / n_total
    if (!is.na(acc) && acc > best_acc) {
      best_acc <- acc
      best_thr <- t
    }
  }
  return(list(threshold = best_thr, accuracy = best_acc))
}


quiet_png <- function(file, expr) {
  png(file, 600, 400)
  on.exit(invisible(capture.output(dev.off())), add = TRUE)
  invisible(capture.output(expr))
}

has_reals <- "0" %in% df$true_label
has_fakes <- "1" %in% df$true_label
selected_thr <- NULL

# ── 4. Determine Threshold ──────────────────────────────────────────────────
if (!is.null(args$`use-threshold`)) {
    log(sprintf("Using user-defined threshold: %.3f", args$`use-threshold`))
    selected_thr <- args$`use-threshold`
} else if (has_reals && has_fakes) {
    log("Both REAL and FAKE data found. Searching for best F1 threshold.")
    idx_fake <- which(df$true_label == 1)
    idx_real <- which(df$true_label == 0)
    
    thr_vec <- replicate(args$n_iter, {
      sel <- sample(idx_real, min(length(idx_real), length(idx_fake)), replace=TRUE)
      evaluate_once_f1(rbind(df[sel, ], df[idx_fake, ]), args$step)
    })
    selected_thr <- as.numeric(names(sort(table(thr_vec), TRUE)[1]))
    log(sprintf("Modal best-F1 threshold = %.3f", selected_thr))
} else {
    log("Fakes-only data detected. F1 optimization is not applicable. Using 0.5 as a default threshold for reporting.")
    selected_thr <- 0.5
}


# ── 5. Perform Analysis & Generate Outputs ------------------------------------
log("Generating analysis reports and plots...")

sink(paste0("detection_diagnostic_results", file_suffix, ".txt"))
cat("=== Analysis Configuration ===\n")
cat("Timestamp:", format(Sys.time()), "\n")
cat("Selected Threshold for Main Analysis:", selected_thr, "\n")
if(is.null(args$`use-threshold`) && has_reals) cat("Threshold Method: F1 Maximization\n")
if(!is.null(args$`use-threshold`)) cat("Threshold Method: User-Defined\n")
if(!has_reals) cat("Threshold Method: Default (Fakes-Only Mode)\n")
cat("\n")

if (has_reals && has_fakes) {
    # Full two-class analysis
    pred_labels <- df$score >= selected_thr
    roc_obj <- roc(df$true_label, df$score, quiet = TRUE)
    cat("=== AUC ===\n"); print(auc(roc_obj))
    
    best_acc_info <- find_best_accuracy_threshold(df)
    cat("\n=== Optimal Accuracy Analysis ===\n")
    cat("Note: This is calculated independently of the main analysis threshold.\n")
    cat(sprintf(
      "The threshold that maximizes accuracy on the full dataset is %.3f (Accuracy = %.4f)\n",
      best_acc_info$threshold,
      best_acc_info$accuracy
    ))
    
    conf_mat <- table(True = df$true_label, Pred = as.integer(pred_labels))
    cat("\n=== Confusion Matrix (using main analysis thr=", sprintf("%.3f", selected_thr), ") ===\n", sep = ""); print(conf_mat)
    write.csv(as.data.frame.matrix(conf_mat), sprintf("confusion_matrix_thr%.3f%s.csv", selected_thr, file_suffix))

    quiet_png(paste0("roc_curve", file_suffix, ".png"), plot(roc_obj, main = "ROC Curve"))
    quiet_png(paste0("violin_plot_scores", file_suffix, ".png"),
          print(ggplot(df, aes(x = factor(true_label, labels=c('Real','Fake')), y = score, fill=factor(true_label))) +
                  geom_violin() + labs(title="Score Distribution by Class", x="Class", y="Score") + theme_minimal()))

} else if (has_fakes) {
    log("WARNING: Fakes-only data. TN and FP will be zero in the confusion matrix.")
    cat("=== Fakes-Only Analysis ===\n")
    pred_labels <- df$score >= selected_thr
    detection_rate <- mean(pred_labels)
    cat(sprintf("Detection Rate at thr=%.3f: %.2f%%\n", selected_thr, detection_rate * 100))
    cat("\nScore Summary Statistics for Fakes:\n"); print(summary(df$score))
    
    tp <- sum(df$score >= selected_thr)
    fn <- sum(df$score < selected_thr)
    
    conf_mat_data <- c(0, 0, fn, tp) # TN, FP, FN, TP
    conf_mat <- matrix(conf_mat_data, nrow=2, byrow=TRUE)
    colnames(conf_mat) <- c("Predicted_Real", "Predicted_Fake")
    rownames(conf_mat) <- c("Actual_Real", "Actual_Fake")

    cat("\n=== Confusion Matrix (thr=", sprintf("%.3f", selected_thr), ") ===\n", sep = "");
    print(conf_mat)
    write.csv(as.data.frame.matrix(conf_mat), sprintf("confusion_matrix_thr%.3f%s.csv", selected_thr, file_suffix))

    quiet_png(paste0("histogram_fake_scores", file_suffix, ".png"), 
        print(ggplot(df, aes(x=score)) + geom_histogram(bins=50, fill='tomato', alpha=0.7) +
        geom_vline(xintercept=selected_thr, color="blue", linetype="dashed", size=1) +
        labs(title="Distribution of Fake Scores", x="Score", y="Count") +
        annotate("text", x=selected_thr, y=0, label=paste("Threshold =", selected_thr), vjust=-1, hjust=-0.1, color="blue")
    ))
}
sink()

if(nrow(df) > 1 && has_reals) {
    quiet_png(paste0("score_distribution_density", file_suffix, ".png"),
        print(ggplot(df, aes(x = score, color = factor(true_label, labels=c('Real', 'Fake')))) + geom_density(size=1) +
        labs(title="Score Density", x="Score", color="Class") + theme_minimal())
    )
}

log(paste("✓ Analysis complete. Artefacts saved in", out_dir))