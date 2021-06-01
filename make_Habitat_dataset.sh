#!/bin/bash
python Scripts/Tiff2Npy.py bedrockC_240 "Images/Habitats/Bedrock/*/TIFF/*.tif" Results/AE_Darter_bright/ -c 2 -n 240

python Scripts/Tiff2Npy.py boulderC_240 "Images/Habitats/Boulder/*/TIFF/*.tif" Results/AE_Darter_bright/ -c 2 -n 240

python Scripts/Tiff2Npy.py detritusC_240 "Images/Habitats/Detritus/*/TIFF/*.tif" Results/AE_Darter_bright/ -c 2 -n 240

python Scripts/Tiff2Npy.py gravelC_240 "Images/Habitats/Gravel/*/TIFF/*.tif" Results/AE_Darter_bright/ -c 2 -n 240

python Scripts/Tiff2Npy.py sandC_240 "Images/Habitats/Sand/*/TIFF/*.tif" Results/AE_Darter_bright/ -c 2 -n 240
