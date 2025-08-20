#!/usr/bin/env Rscript
# deepfake_detection_threshold_analytics_balanced.R
# -----------------------------------------------------------------------------
# Balanced evaluation for DeepMedia detector scores (CLI-friendly).
# [VERSION: FINAL ROBUST 2025-08-11]
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse);  library(pROC);  library(ROCR)
  library(ggplot2);   library(ResourceSelection)
})



# ── 1. CLI -------------------------------------------------------------------
opt <- OptionParser(option_list = list(
  make_option("--single"), make_option("--fake"),
  make_option("--real"),   make_option("--out", default = "."),
  make_option("--n_iter",  type = "integer", default = 200),
  make_option("--step",    type = "double",  default = 0.01),
  make_option("--anchors", default = "0.30,0.50,0.70"),
  make_option("--no-balance", action="store_true", default=FALSE, help="Skip balanced subsampling and use the full dataset.")
))
args <- parse_args(opt)
set.seed(42)

anchor_thr <- as.numeric(strsplit(args$anchors, ",")[[1]])
out_dir    <- normalizePath(args$out, mustWork = FALSE)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
setwd(out_dir)

log <- function(msg) message(sprintf("[%s] %s", format(Sys.time(), "%H:%M:%S"), msg))

# ── 2. Load data -------------------------------------------------------------
log("Loading detector CSV(s)")
if (!is.null(args$single)) {
  df <- read.csv(args$single); src <- args$single
  if (!"true_label" %in% names(df)) stop("single CSV must contain true_label")
} else if (!is.null(args$fake) && !is.null(args$real)) {
  df_fake <- read.csv(args$fake); df_fake$true_label <- 1
  df_real <- read.csv(args$real); df_real$true_label <- 0
  df <- rbind(df_fake, df_real); src <- c(args$fake, args$real)
} else stop("Provide either --single or both --fake & --real")

log(paste("Loaded", nrow(df), "rows from", paste(src, collapse = ", ")))
if (!"score" %in% names(df)) stop("Need a 'score' column")

# ── 3. Helper functions ------------------------------------------------------
evaluate_once <- function(df_in, step = 0.01) {
  best_f1 <- -Inf; best_thr <- NA
  for (t in seq(0,1,by=step)) {
    prd <- df_in$score >= t
    tp  <- sum(prd & df_in$true_label == 1)
    fp  <- sum(prd & df_in$true_label == 0)
    fn  <- sum(!prd & df_in$true_label == 1)
    prec <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
    rec  <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
    f1   <- ifelse(is.na(prec) | is.na(rec) | (prec + rec == 0),
                   NA,
                   2 * prec * rec / (prec + rec))
    if (!is.na(f1) && f1 > best_f1) {
      best_f1 <- f1
      best_thr <- t
    }
  }
  best_thr
}

quiet_png <- function(file, expr) {
  png(file, 600, 400)
  on.exit(invisible(capture.output(dev.off())), add = TRUE)
  invisible(capture.output(expr))
}

# ── 4. Conditional subsampling -----------------------------------------------
idx_fake <- which(df$true_label == 1)
idx_real <- which(df$true_label == 0)
n_fake <- length(idx_fake)
n_real <- length(idx_real)

if (!args$`no-balance`) {
  log("Searching modal best-F1 threshold with balanced subsampling")
  
  if (n_fake == 0 || n_real == 0) {
    stop("Both fake and real classes must have at least one sample for balanced sampling.")
  }
  
  sample_size <- min(n_fake, n_real)
  log(paste("Subsampling to", sample_size, "records per class for each of", args$n_iter, "iterations."))

  thr_vec <- replicate(args$n_iter, {
    sel_fake <- sample(idx_fake, sample_size)
    sel_real <- sample(idx_real, sample_size)
    evaluate_once(rbind(df[sel_fake, ], df[sel_real, ]), args$step)
  })
  modal_thr <- as.numeric(names(sort(table(thr_vec), TRUE)[1]))

} else {
  log("Skipping balanced subsampling as per --no-balance flag.")
  log("Finding best F1 threshold on the full, unbalanced dataset.")
  
  modal_thr <- evaluate_once(df, args$step)
  if (is.na(modal_thr)) {
    stop("Could not determine a valid F1 threshold on the full dataset.")
  }
  thr_vec <- c(modal_thr)
}

log(sprintf("Selected threshold = %.2f", modal_thr))


# ── 5. Classical stats (NOW WITH ROBUST ERROR HANDLING) ----------------------
log("Computing statistics on full dataset")

ks_result <- tryCatch({
  ks.test(df$score[df$true_label == 0], df$score[df$true_label == 1], exact = TRUE)
}, error = function(e) {
  log(paste("Skipping KS Test due to error:", e$message))
  return(paste("Skipped due to error:", e$message))
})

wilcox_result <- tryCatch({
  wilcox.test(df$score[df$true_label == 0], df$score[df$true_label == 1])
}, error = function(e) {
  log(paste("Skipping Wilcoxon Test due to error:", e$message))
  return(paste("Skipped due to error:", e$message))
})

roc_obj       <- roc(df$true_label, df$score, quiet = TRUE)
auc_val       <- auc(roc_obj)
model         <- glm(true_label ~ score, family = "binomial", data = df)
pred_rocr <- ROCR::prediction(df$score, df$true_label)
pred_modal <- df$score >= modal_thr
TrueFac    <- factor(df$true_label, levels = c(0, 1))
PredFac    <- factor(ifelse(pred_modal, 1, 0), levels = c(0, 1))
conf_mat   <- table(True = TrueFac, Pred = PredFac)

# ── 6. Save diagnostics & summaries -----------------------------------------
log("Writing diagnostics files")

# Diagnostic text file
sink("detection_diagnostic_results.txt"); {
  cat("=== KS Test ===\n");    print(ks_result)
  cat("\n=== Wilcoxon Test ===\n"); print(wilcox_result)
  cat("\n=== Logistic Regression ===\n"); print(summary(model))
  cat("\n=== AUC ===\n");      print(auc_val)
  # EER calculation
  tryCatch({
      perf_det  <- ROCR::performance(pred_rocr, "fnr", "fpr")
      fpr_det   <- perf_det@x.values[[1]]
      fnr_det   <- perf_det@y.values[[1]]
      eer_idx   <- which.min(abs(fpr_det - fnr_det))
      eer_val   <- mean(c(fpr_det[eer_idx], fnr_det[eer_idx]))
      cat("\n=== EER ===\n", eer_val, "\n")
  }, error = function(e) {
      cat("\n=== EER ===\nSkipped due to error.\n")
  })
  cat("\n=== Confusion Matrix (thr=", sprintf("%.2f", modal_thr), ") ===\n", sep = "")
  print(conf_mat)
}; sink()

# Threshold summary text file
sink("detection_threshold_summary.txt"); {
  cat("### Selected threshold across", args$n_iter, "runs\n\n")
  cm   <- conf_mat
  if ((cm["1", "1"] + cm["0", "1"]) > 0) {
    prec <- cm["1", "1"] / (cm["1", "1"] + cm["0", "1"])
  } else {
    prec <- 0
  }
  if ((cm["1", "1"] + cm["1", "0"]) > 0) {
    rec <- cm["1", "1"] / (cm["1", "1"] + cm["1", "0"])
  } else {
    rec <- 0
  }
  if ((prec + rec) > 0) {
    f1 <- 2 * prec * rec / (prec + rec)
  } else {
    f1 <- 0
  }
  cat(sprintf("thr = %.2f | Precision = %.3f | Recall = %.3f | F1 = %.3f\n\n",
              modal_thr, prec, rec, f1))
  cat("--- Top thresholds ---\n"); print(head(sort(table(thr_vec), TRUE), 10))
}; sink()

write.csv(as.data.frame.matrix(conf_mat),
          sprintf("confusion_matrix_modalThr%.2f.csv", modal_thr),
          row.names = TRUE)

# ── 7. Plots (NOW WITH ROBUST ERROR HANDLING) --------------------------------
log("Generating plots")

tryCatch({
  quiet_png("roc_curve.png", plot(roc_obj, main = "ROC Curve"))
}, error = function(e) { log("Skipping roc_curve.png due to error.") })

tryCatch({
  perf_det  <- ROCR::performance(pred_rocr, "fnr", "fpr")
  fpr_det   <- perf_det@x.values[[1]]
  fnr_det   <- perf_det@y.values[[1]]
  quiet_png("det_curve.png", {
    plot(qnorm(fpr_det), qnorm(fnr_det), type = "l",
         main = "DET Curve", xlab = "FPR (probit)", ylab = "FNR (probit)")
    abline(0, 1, lty = 2)
  })
}, error = function(e) { log("Skipping det_curve.png due to error.") })

tryCatch({
  pr_perf <- ROCR::performance(pred_rocr, "prec", "rec")
  quiet_png("pr_curve.png",
            plot(pr_perf@x.values[[1]], pr_perf@y.values[[1]], type = "l",
                 xlab = "Recall", ylab = "Precision", main = "Precision-Recall Curve"))
}, error = function(e) { log("Skipping pr_curve.png due to error.") })

tryCatch({
  if (length(unique(df$true_label)) > 1) {
      quiet_png("violin_plot.png",
            print(ggplot(df, aes(factor(true_label, labels = c("Real", "Fake")), score)) +
                    geom_violin(fill = "skyblue") +
                    geom_boxplot(width = .1, outlier.shape = NA) +
                    theme_minimal() + 
                    labs(x = "Class", y = "Score")))
  } else {
      log("Skipping violin plot: only one class present in the data.")
  }
}, error = function(e) { log("Skipping violin_plot.png due to error.") })

tryCatch({
  bins <- cut(df$score, breaks = seq(0, 1, 0.1), include.lowest = TRUE)
  cal_df <- data.frame(pred = tapply(df$score, bins, mean, na.rm=TRUE),
                       true = tapply(df$true_label, bins, mean, na.rm=TRUE))
  cal_df <- na.omit(cal_df)
  quiet_png("calibration_curve.png", {
    print(ggplot(cal_df, aes(pred, true)) +
            geom_point(size = 3) +
            geom_abline(slope = 1, intercept = 0, lty = 2) +
            theme_minimal() +
            labs(x = "Mean predicted", y = "Empirical"))
  })
}, error = function(e) { log("Skipping calibration_curve.png due to error.") })


# ── 8. Finish ----------------------------------------------------------------
log("✓ Analysis complete")
log(paste("  Artefacts saved in", out_dir))