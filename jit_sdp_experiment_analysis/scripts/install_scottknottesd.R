if (!requireNamespace("remotes", quietly = TRUE)) {
  stop("The R package 'remotes' is required. Install it with conda package r-remotes.")
}

remotes::install_github("klainfo/ScottKnottESD", ref = "development", upgrade = "never")

if (!requireNamespace("ScottKnottESD", quietly = TRUE)) {
  stop("ScottKnottESD installation failed.")
}

packageVersion("ScottKnottESD")
