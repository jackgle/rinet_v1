library(refineR)

# parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript run_refineR.R <ribench_data_dir> [output_dir]")
}
indir <- args[1]
if (length(args) >= 2) {
    outdir <- args[2]
} else {
    script_dir <- dirname(normalizePath(sub('--file=', '', grep('--file=', commandArgs(trailingOnly = FALSE), value = TRUE)[1])))
    outdir <- file.path(script_dir, 'refineR_predictions')
}

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# collect all CSV files
analyte_dirs <- list.dirs(indir, recursive = FALSE, full.names = TRUE)

files_test <- c()
for (adir in analyte_dirs) {
    files_test <- c(files_test, list.files(adir, full.names = TRUE, pattern = '\\.csv$'))
}

files_test <- sample(files_test)
cat('Number of files:', length(files_test), '\n')
cat('Predicting\n')

pb <- txtProgressBar(min = 0, max = length(files_test), style = 3)

i <- 0
for (file in files_test) {

    i <- i + 1

    outfile <- basename(file)

    if (!file.exists(file.path(outdir, outfile))) {

        data <- read.csv(file, header=FALSE)$V1

        # Use modBoxCox model for LDH, default for other analytes
        if (grepl("_LDH_", outfile)) {
            fit <- findRI(data, model = "modBoxCox")
        } else {
            fit <- findRI(data)
        }

        # CRP: use 95th percentile (single upper limit); others: default 2.5/97.5
        if (grepl("_CRP_", outfile)) {
            result <- getRI(fit, RIperc = c(0.05, 0.95), Scale='original')
        } else {
            result <- getRI(fit, Scale='original')
        }

        write.csv(result, file.path(outdir, outfile))
    }

    setTxtProgressBar(pb, i)
}

close(pb)
