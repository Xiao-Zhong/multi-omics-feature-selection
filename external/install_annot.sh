#!/usr/bin/env bash
export R_LIBS_USER="/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/Rlib"
export TMPDIR="/mnt/data/hackathon/xiao/mpm_multiomics_pipeline/external/tmp"
mkdir -p "$R_LIBS_USER" "$TMPDIR"
Rscript -e '
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
options(timeout=3600)
BiocManager::install(c("hgu133plus2cdf","hgu133plus2.db"),
                     lib=Sys.getenv("R_LIBS_USER"), update=FALSE, ask=FALSE)
for(p in c("AnnotationDbi","org.Hs.eg.db","hgu133plus2cdf","hgu133plus2.db"))
  cat(p, requireNamespace(p, quietly=TRUE), "\n")
' 2>&1
echo "ANNOT_INSTALL_DONE"
