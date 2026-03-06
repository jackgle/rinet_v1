library(RIbench)

dr = './RIbench/'

# generate simulated test sets
generateBiomarkerTestSets(workingDir=dr)

# get test set metadata including ground truth RIs
test_meta <- loadTestsetDefinition()
write.csv(test_meta, paste0(dr,'BMTestSets_meta.csv'))
