#!/bin/sh

spec_perworker=`expr $2 / $1`
echo $spec_perworker
for i in $(seq 1 $1)
do
    if [ $i = $1 ]
    then
        echo "$i $(( $spec_perworker * (i - 1) )) $2"
    else
        echo "$i $(( $spec_perworker * (i - 1) )) $(( $spec_perworker * i ))"
    fi
done

