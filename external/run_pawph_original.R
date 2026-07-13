## Run ORIGINAL PAWPH (r08in/PAWPH) on the MESOMICS pool, compare to in-house sel_pawph.
.libPaths(c(file.path(getwd(), "Rlib"), .libPaths()))
suppressMessages({library(survival); library(MASS); library(gbm); library(ncvreg2)})
setwd("/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external")
source("PAWPH/prcox.R")

X <- as.matrix(read.csv("pawph_X.csv", row.names = 1, check.names = FALSE))
sv <- read.csv("pawph_surv.csv", row.names = 1)
y <- Surv(pmax(sv$months, 1e-3), sv$event)
cat(sprintf("[data] %d samples x %d features, %d events\n", nrow(X), ncol(X), sum(sv$event)))

## PAWPH is designed for modest p; run on the full 800-pool but cap lambda grid & folds for tractability.
set.seed(42)
t0 <- Sys.time()
res <- tryCatch(
  prcoxreg(y, X, seed = 42, alpha = 0.5, pout.max = 0.2, nlambda = 30, nfolds = 5, lambda.min = 0.1),
  error = function(e) { cat("[prcoxreg ERROR]", conditionMessage(e), "\n"); NULL })
cat(sprintf("[time] %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
if (is.null(res)) quit(status = 1)

beta <- res$betaHat_re                      # final refit coefficients (the PAWPH estimator)
names(beta) <- colnames(X)
sel <- beta[beta != 0]
imp <- sort(abs(sel), decreasing = TRUE)
cat(sprintf("[pawph] selected %d features (nonzero beta); opt.alpha=%s\n",
            length(sel), as.character(res$opt.alpha)))
n_out <- sum(res$gammaHat != 0)
cat(sprintf("[pawph] flagged %d/%d samples as outliers (nonzero gamma)\n", n_out, nrow(X)))
panel <- names(imp)[1:min(20, length(imp))]
cat("[pawph] top panel:\n"); print(panel)

write.csv(data.frame(feature = names(imp), abs_beta = as.numeric(imp)),
          "pawph_original_importance.csv", row.names = FALSE)
writeLines(jsonlite_ok <- panel, "pawph_original_panel.txt")
cat("[done] wrote pawph_original_importance.csv + pawph_original_panel.txt\n")
