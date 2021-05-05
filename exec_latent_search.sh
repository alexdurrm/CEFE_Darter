#!/bin/bash
middle=("Bedrock" "Boulder" "Detritus" "Gravel" "Sand")
networks=("ConvolutionnalModel" "Perceptron" "SparseConvolutionnal" "VariationnalAE" "VGG16AE")

for i in ${middle[@]};
do
	for j in ${networks[@]};
	do
		python Scripts/AutoEncoders/${j}.py LD_selection Images/Habitats/${i}/Train_${i}240_presizeNonexNone_L3_pred128x128x3.npy Images/Habitats/${i}/Test_${i}240_presizeNonexNone_L3_pred128x128x3.npy -b 25
	done
done
