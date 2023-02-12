#!/bin/sh

spec_perworker=`expr $2 / $1`
for i in $(seq 1 $1)
do
    if [ $i = $1 ]
    then
        nohup python genova_preprocess.py $i $(( $spec_perworker * (i - 1) )) $2 &
    else    
        nohup python genova_preprocess.py $i $(( $spec_perworker * (i - 1) )) $(( $spec_perworker * i )) &
    fi
done