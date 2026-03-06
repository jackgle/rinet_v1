library(refineR)

indir <- './test_csvs/'
outdir <- './refineR_predictions/'

dir.create(outdir, showWarnings = FALSE)

cat('Reading file list\n')
files_test <- list.files(indir)
files_test <- sample(files_test)
cat('Number of files:', length(files_test), '\n')

cat('Predicting\n')

pb <- txtProgressBar(min = 0, max = length(files_test), style = 3)

i <- 0
for (file in files_test) {

    i <- i + 1

    if (!file.exists(paste0(outdir, file))) {

        data <- read.csv(paste0(indir, file), header=FALSE)$V1
        fit <- findRI(data)
        result <- getRI(fit, Scale='original')

        write.csv(result, paste0(outdir, file))
    }

    setTxtProgressBar(pb, i)
}

close(pb)
