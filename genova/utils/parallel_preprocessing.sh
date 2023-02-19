#!/bin/sh

for i in $(seq 1 $1)
do
    nohup python genova_preprocess.py $i $1 $2 $3 &
done