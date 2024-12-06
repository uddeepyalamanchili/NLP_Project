#!/bin/bash

set -e
set -x

# !git clone "https://github.com/s-ankur/wikiextract"

cd wikiextract/
     
# Run the setup file. It downloads the required repos and data dumps
bash setup.sh 
     
# You need to run these two steps after every failed run. Because we use these to track the progress.
rm hindi.output hindi-pos-tagger-3.0/hindi.input.txt
rm extracted
     
# This step is used to run the actual code. it might take some time to run.
bash run_hiwiki.sh

# A sample of the generated sentences
head -4000 hiwiki.augmented.edits|python scripts/convert_to_wdiff.py|shuf -n 40

wc hiwiki.augmented.edits

# Execute the executionFile.py in the wikiextract directory and update the count values from the output
head -7823271 hiwiki.augmented.edits>train
tail -1564656 hiwiki.augmented.edits>val
mkdir -p data


# Execute the executionFile2.py in the wikiextract directory before this

rm train val
bash shuffle.sh data/train_merge 42
bash shuffle.sh data/valid 42     
wc data/*
mv data ../data
