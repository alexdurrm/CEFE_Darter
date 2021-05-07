#!/bin/bash
python Scripts/Utils/Tiff2Npy.py Sand "Images/Habitats/Sand/*/TIFF/*.tif" Images/Habitats/Sand/ -n 240
python Scripts/Utils/Tiff2Npy.py Gravel "Images/Habitats/Gravel/*/TIFF/*.tif" Images/Habitats/Gravel/ -n 240
python Scripts/Utils/Tiff2Npy.py Detritus "Images/Habitats/Detritus/*/TIFF/*.tif" Images/Habitats/Detritus/ -n 240
python Scripts/Utils/Tiff2Npy.py Boulder "Images/Habitats/Boulder/*/TIFF/*.tif" Images/Habitats/Boulder/ -n 240
python Scripts/Utils/Tiff2Npy.py Bedrock "Images/Habitats/Bedrock/*/TIFF/*.tif" Images/Habitats/Bedrock/ -n 240
