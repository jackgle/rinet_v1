library(refineR)

# resolve paths relative to this script's location
script_dir <- dirname(normalizePath(sub('--file=', '', grep('--file=', commandArgs(trailingOnly = FALSE), value = TRUE)[1])))
indir <- file.path(script_dir, '..', '..', 'data', 'RIbench', 'Data')
outdir <- file.path(script_dir, 'refineR_predictions')

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
        fit <- findRI(data)
        result <- getRI(fit, Scale='original')

        write.csv(result, file.path(outdir, outfile))
    }

    setTxtProgressBar(pb, i)
}

close(pb)
