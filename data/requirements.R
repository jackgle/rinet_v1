cat("Starting install...\n")
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(timeout = 300)

install.packages("RIbench", quiet = FALSE)

cat("Done.\n")