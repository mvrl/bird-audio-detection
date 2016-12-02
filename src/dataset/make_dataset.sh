#!/bin/bash
BASE=../../data/

# assumes it is already downloaded.  CSV files are in $BASE and audio
# files are in $BASE/wav

# remove header, randomly shuffle, split into 10 roughly equal chunks,
# save output
RAWFILE=$BASE/freefield1010_labels.csv

tail -n +2 $RAWFILE | \
  shuf | \
  split -d -l $[ $(wc -l $RAWFILE | cut -d" " -f1) * 10 / 100 ] - freefield1010_

sed -i -e 's/^/freefield1010_audio\/wav\//' ./freefield1010_*

RAWFILE=$BASE/warblr_labels.csv

tail -n +2 $RAWFILE | \
  shuf | \
  split -d -l $[ $(wc -l $RAWFILE | cut -d" " -f1) * 10 / 100 ] - warblr_

sed -i -e 's/^/warblr_audio\/wav\//' ./warblr_*
