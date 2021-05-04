#!/bin/bash

species=("barrenense" "caeruleum" "camurum" "chlorosomum" "gracile" "olmstedi" "punctulatum" "pyrrhogaster" "swaini" "zonale" "zonistium")

for i in ${species[@]};
do
	python Scripts/AutoEncoders/ConvolutionnalModel.py Images/Habitats/${i} Train_${i}_presize600x900_pred128x128x3.npy Images/Habitats/${i} Test_${i}_presize600x900_pred128x128x3.npy -b 25

	python Scripts/AutoEncoders/Perceptron.py Images/Habitats/${i}/Train_${i}_presize600x900_pred128x128x3.npy Images/Habitats/${i}/Test_${i}_presize600x900_pred128x128x3.npy -b 25 -l 200

done
