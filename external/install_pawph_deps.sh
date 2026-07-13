#!/usr/bin/env bash
cd /mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external
export R_LIBS_USER="$PWD/Rlib"; mkdir -p "$R_LIBS_USER"
export TMPDIR="$PWD/tmp"; mkdir -p "$TMPDIR"
Rscript -e 'options(repos=c(CRAN="https://cloud.r-project.org")); install.packages(c("Rcpp","gbm"), lib=Sys.getenv("R_LIBS_USER"))' 2>&1
R CMD INSTALL -l "$R_LIBS_USER" PAWPH/packages/ncvreg2_3.13.0.tar.gz 2>&1
Rscript -e '.libPaths(Sys.getenv("R_LIBS_USER")); for(p in c("gbm","ncvreg2","survival","MASS")) cat(p, requireNamespace(p, quietly=TRUE), "\n")' 2>&1
echo "INSTALL_DONE"
