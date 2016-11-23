#!/bin/bash
BASE=~/data/birddetection

# assumes it is already downloaded.  CSV files are in $BASE and audio
# files are in $BASE/wav

# remove header, randomly shuffle, split into 10 roughly equal chunks,
# save output
RAWFILE=$BASE/ff1010bird_metadata.csv

tail -n +2 $RAWFILE | \
  shuf | \
  split -d -l $[ $(wc -l $RAWFILE | cut -d" " -f1) * 10 / 100 ] - ff1010bird_

RAWFILE=$BASE/warblrb10k_public_metadata.csv

tail -n +2 $RAWFILE | \
  shuf | \
  split -d -l $[ $(wc -l $RAWFILE | cut -d" " -f1) * 10 / 100 ] - warblrb10k_

