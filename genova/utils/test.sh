#!/bin/sh

for i in {1..$1}
do
    if [ $i = $1 ]
    then
        nohup python genova_preprocess.py $i $[ 61314 * ($i - 1) ] 1962054 &
    else    
        nohup python genova_preprocess.py $i $[ 61314 * ($i - 1) ] $[ 61314 * $i ] &
    fi
done