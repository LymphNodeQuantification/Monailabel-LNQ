#!/bin/bash

CTS=/mnt/sdp-timc-correction/timc-ln10-2022-11-11/ct-subset
CORRECTIONS=/mnt/sdp-timc-correction/timc-ln10-2022-11-11/segmentations-corrected-nii/

RETRAIN=/mnt/data/datasets/tcia-ln10-retrain-2022-12/imagesTr
RETRAIN_LABELS=${RETRAIN}/labels/final

mkdir -p ${RETRAIN}
mkdir -p ${RETRAIN_LABELS}

for label in ${CORRECTIONS}/*.gz ; do ln -s ${label} ${RETRAIN_LABELS} ; done

for ct in ${CTS}/*.gz ; do ln -s ${ct} ${RETRAIN}; done


