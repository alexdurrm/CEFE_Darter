#!/bin/bash

#parameters
DIR_RESULTS="Results/NST_metrics/"


#list images
python Scripts/Metrics/list_images.py both Images/StyleTransferImages/ -o $DIR_RESULTS -f .jpg .png -e True

#deep features vgg16 and vgg19
python Scripts/Metrics/DeepFeatureMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t RGB -c ALL -m vgg16
python Scripts/Metrics/DeepFeatureMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t RGB -c ALL -m vgg19

#fft slopes
python Scripts/Metrics/FFTMetrics.py work slope ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t DARTER -c GRAY -d 512

#Gabor filters
python Scripts/Metrics/GaborMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t DARTER -c GRAY

#Haralick descriptors
python Scripts/Metrics/GLCMMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t DARTER -c GRAY 

#PHOG 
python Scripts/Metrics/PHOGMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULT

#LBPHistMetrics
python Scripts/Metrics/LBPMetrics.py work lbp ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -b 256
python Scripts/Metrics/LBPMetrics.py work best_lbp ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -b 100

#statistical features
python Scripts/Metrics/ScalarMetrics.py work c_ratio ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t DARTER -c ALL
python Scripts/Metrics/ScalarMetrics.py work moments ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t DARTER -c GRAY

echo "DONE"
