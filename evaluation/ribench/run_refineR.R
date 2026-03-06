install.packages("refineR", repos = "https://cloud.r-project.org")
library(refineR)

indr = './ribench_sample/'
outdr = './refineR_predictions/'

dir.create(outdr)

cat('Reading file list\n')
files_test <- read.csv(paste0(indr,'files.csv'))$file
files_test <- sample(files_test)
cat('Number of files:', length(files_test), '\n')

cat('Predicting\n')

pb <- txtProgressBar(min = 0, max = length(files_test), style = 3)

i <- 0
for (file in files_test) {

    i <- i + 1

    split_string <- strsplit(file, "/")[[1]]
    outfile <- tail(split_string, 1)

    if (!file.exists(paste0(outdr, outfile))) {

        data <- read.csv(file, header=FALSE)$V1
        fit <- findRI(data)
        result <- getRI(fit, Scale='original')

        write.csv(result, paste0(outdr, outfile))
    }

    setTxtProgressBar(pb, i)
}

close(pb)