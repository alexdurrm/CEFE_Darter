#!/bin/bash
species=("barrenense" "blennioides" "caeruleum" "camurum" "extra" "gracile" "olmestedi" "punctulatum" "pyrrhogaster" "swaini")

for i in ${species[@]};
do
	echo python Scripts/Metrics/AutoencoderMetrics.py work Results_AE/image_list.csv Scripts/AutoEncoders/TrainedModels/Perceptron_${i}_LD200_pred_128x128x3/
	python Scripts/Metrics/AutoencoderMetrics.py work Results_AE/image_list.csv Scripts/AutoEncoders/TrainedModels/Perceptron_${i}_LD200_pred128x128x3/

	#python Scripts/Metrics/AutoencoderMetrics.py work Results_AE/image_list.csv Scripts/AutoEncoders/TrainedModels/Convolutional_${i}_LD20_pred_128x128x3/	
done
