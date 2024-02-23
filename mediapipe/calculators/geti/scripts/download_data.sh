#!/bin/bash

DATA_FOLDER="/data"
REPOSITORY_PATH="https://repository.toolbox.iotg.sclab.intel.com/mediapipe"
REPOSITORY_MODEL_PATH="$REPOSITORY_PATH/models/"
REPOSITORY_DATA_PATH="$REPOSITORY_PATH/data/"
REPOSITORY_REFERENCE_PATH="$REPOSITORY_PATH/reference/"

MODELS=(
   "intel/face-detection-retail-0004/face-detection-retail-0004.bin"
   "intel/face-detection-retail-0004/face-detection-retail-0004.xml"
   "public/efficientnet-b0/efficientnet-b0-pytorch.bin"
   "public/efficientnet-b0/efficientnet-b0-pytorch.xml"
   "public/hrnet-v2-c1-segmentation/hrnet-v2-c1-segmentation.bin"
   "public/hrnet-v2-c1-segmentation/hrnet-v2-c1-segmentation.xml"
   "public/ssd-resnet34-1200-onnx/ssd-resnet34-1200-onnx.bin"
   "public/ssd-resnet34-1200-onnx/ssd-resnet34-1200-onnx.xml"
   "public/anomaly_stfpm_bottle_mvtec/anomaly_stfpm_bottle_mvtec.bin"
   "public/anomaly_stfpm_bottle_mvtec/anomaly_stfpm_bottle_mvtec.xml"
)

GETI_MODELS=(
   "anomaly_classification_padim"
   "anomaly_detection_padim"
   "anomaly_segmentation_padim"
   "classification_efficientnet_b0"
   "detection_atss"
   "detection_ssd"
   "detection_yolox"
   "instance_segmentation_maskrcnn_efficientnet_b2b"
   "instance_segmentation_maskrcnn_resnet50"
   "rotated_detection_maskrcnn_resnet50"
   "rotated_detection_maskrcnn_resnet50_tiling"
   "segmentation_lite_hrnet_18"
   "segmentation_lite_hrnet_18_mod2"
   "segmentation_lite_hrnet_s_mod2"
   "segmentation_lite_hrnet_x_mod3"
)

GETI_REFERENCE=(
   "anomaly_classification_padim.json"
   "anomaly_detection_padim.json"
   "anomaly_segmentation_padim.json"
   "classification_efficientnet_b0.json"
   "detection_atss.json"
   "detection_ssd.json"
   "detection_yolox.json"
   "instance_segmentation_maskrcnn_efficientnet_b2b.json"
   "instance_segmentation_maskrcnn_resnet50.json"
   "rotated_detection_maskrcnn_resnet50.json"
   "segmentation_lite_hrnet_18.json"
   "segmentation_lite_hrnet_18_mod2.json"
   "segmentation_lite_hrnet_s_mod2.json"
   "segmentation_lite_hrnet_x_mod3.json"
)

DATASET=(
   "000000000074.jpg"
   "pearl.jpg"
   "cattle.jpg"
)

# Check if data folder is accessible
if [ -d $DATA_FOLDER ]
then
   echo "Data folder: '$DATA_FOLDER' exists"
else
   mkdir -vp $DATA_FOLDER
fi

for i in "${DATASET[@]}"
do
   URL="$REPOSITORY_DATA_PATH$(basename "$i")"
   HTTPS_PROXY='' wget -N --no-check-certificate $URL -P $DATA_FOLDER
done

for i in "${MODELS[@]}"
do
   URL="$REPOSITORY_MODEL_PATH$(basename "$i")"
   FOLDER="$DATA_FOLDER/omz_models/$(dirname "$i")"
   HTTPS_PROXY='' wget -N --no-check-certificate $URL -P $FOLDER
done

for i in "${GETI_MODELS[@]}"
do
   URL="${REPOSITORY_MODEL_PATH}geti/$i"
   FOLDER="$DATA_FOLDER/geti"
   HTTPS_PROXY='' wget -N --no-check-certificate "$URL.xml" -P $FOLDER
   HTTPS_PROXY='' wget -N --no-check-certificate "$URL.bin" -P $FOLDER
done

for i in "${GETI_REFERENCE[@]}"
do
   URL="${REPOSITORY_REFERENCE_PATH}$i"
   FOLDER="$DATA_FOLDER/geti/reference"
   HTTPS_PROXY='' wget -N --no-check-certificate "$URL" -P $FOLDER
done

echo "Done."
