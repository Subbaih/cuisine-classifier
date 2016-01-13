#!/bin/sh
n=600;
k=5;
sh run_setup.sh $n $k
sh run_megam_features.sh $n $k "['ingredients_unigram']" > ingredients_unigram.log 2>&1
sh run_megam_features.sh $n $k "['prep_unigram']" > prep_unigram.log 2>&1
sh run_megam_features.sh $n $k "['prep_trim_by_pos']" > prep_trim_by_pos.log 2>&1
sh run_megam_features.sh $n $k "['ingredients_unigram','prep_unigram']" > ingredients_unigram_prep_unigram.log 2>&1
sh run_megam_features.sh $n $k "['ingredients_unigram','prep_trim_by_pos']" > ingredients_unigram_prep_trim_by_pos.log 2>&1
sh run_megam_features.sh $n $k "['ingredients_unigram','prep_unigram','prep_trim_by_pos']" > ingredients_unigram_prep_unigram_prep_trim_by_pos.log 2>&1
mkdir -p "$k"fold
mv *.log "$k"fold
