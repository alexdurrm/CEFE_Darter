#!/bin/bash
species=("Boulder" "Bedrock" "Detritus" "Gravel" "Sand")

networks=("ConvolutionnalModel" "SparseConvolutionnal" "VariationnalAE" "VGG16AE")  #Perceptron

for j in ${networks[@]};
do
	for i in ${species[@]};
	do
		python Scripts/AutoEncoders/${j}.py training -l 6 Images/Habitats/${i}/Train_${i}240_presizeNonexNone_L3_pred128x128x3.npy Images/Habitats/${i}/Test_${i}240_presizeNonexNone_L3_pred128x128x3.npy -e 50 -n ${j}Trained

		python Scripts/Metrics/AutoencoderMetrics.py work Results/AE_tif_mixed_bright/image_list.csv Results/AE_tif_mixed_bright/${j}Trained_${i}240_LD6_pred128x128x3/${j}Trained_${i}240_LD6_pred128x128x3/ -o Results/AE_tif_mixed_bright/${j}Trained_${i}240_LD6_pred128x128x3/ -n True -s True
	done
done
