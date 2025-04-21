#!/bin/bash -l


# frequency_64000
for tok in \
    frequency_64000 \
    fw57M_Entropy_frequency-mean-post-merge_64000 \
    fw57M_Entropy_min-mean-post-merge_64000 \
    fw57M_Surprisal_frequency-mean-post-merge_64000 \
    fw57M_Surprisal_min-mean-post-merge_64000
do
    echo "Tokenizing with $tok"
    uv run cli.py data finewebedu-tokenize --subfolder "$tok"
done