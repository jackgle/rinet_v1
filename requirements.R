cat("Starting install...\n")
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(timeout = 300)

install.packages(c("RIbench", "refineR"))

cat("Done.\n")