#!/bin/bash
middle=("Bedrock" "Boulder" "Detritus" "Gravel" "Sand")
networks=("ConvolutionnalModel" "Perceptron" "SparseConvolutionnal" "VariationnalAE" "VGG16AE")

for i in ${middle[@]};
do
	for j in ${networks[@]};
	do
		python Scripts/AutoEncoders/${j}.py Images/Habitats/${i} Train_${i}_presize600x900_pred128x128x3.npy Images/Habitats/${i} Test_${i}_presize600x900_pred128x128x3.npy -b 25 LD_selection
	done
done
