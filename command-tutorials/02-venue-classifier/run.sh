rm -rf command-tutorials/02-venue-classifier/output

allennlp train command-tutorials/02-venue-classifier/config.json -s command-tutorials/02-venue-classifier/output --include-package command-tutorials.02-venue-classifier