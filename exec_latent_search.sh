#!/bin/bash

#script used to test different latent dimensions on different networks

middle=("Bedrock" "Boulder" "Detritus" "Gravel" "Sand")
networks=("ConvolutionnalModel" "SparseConvolutionnal" "VariationnalAE" "VGG16AE")  #Perceptron

for i in ${middle[@]};
do
	for j in ${networks[@]};
	do
		python Scripts/AutoEncoders/${j}.py LD_selection -l 2 4 6 8 10 Images/Habitats/${i}/Train_${i}240_presizeNonexNone_L3_pred128x128x3.npy Images/Habitats/${i}/Test_${i}240_presizeNonexNone_L3_pred128x128x3.npy -b 25
	done
done
