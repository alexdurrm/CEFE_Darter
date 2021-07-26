# setup
To execute the scripts correctly you should launch them from the root directory of the project or export the root directory path with the following command line (for linux).
```
export PYTHONPATH="absolute/path/to/CEFE/"
```
The project depends on a some python libraries (tensorflow, sklearn, numpy, pandas, matplotlib, seaborn...), you can either install them  manually or you can use the conda_list.txt file containing the exact libraries and versions used for development. To do so you can use the following line:
```
conda env create --name <envname> --file conda_list.txt
```
where <envname> is the name of your new environment.

# CEFE_Darter
**_Scripts/_** contains all the Scripts from this project.
    
**_Scripts/Metrics/_** contains the python scripts used to retrieve metrics from images byt the MotherMetrics script.
    
**_Scripts/Utils/_** contains files with common functions for the rest of the project.
    
**_Scripts/Notebooks/_** contains jupyter notebooks, those are for visualisation purposes.
    
**_Scripts/AutoEncoders/_** contains scripts relative to Autoencoders. 

## Scripts/MotherMetric

This file contains multiple classes inheriting from a mother class __"MotherMetric"__. 
All classes in that file inherit from the the mother class and they each take care of a specific type of metric.
All those classes have the following functions:
* function(image): which take an image as input and execute the metric on one image.
* metric_from_path_list(path_list): which allows to execute the metric on a list of images given their path.
* save(path): which save the list of metric values as a csv, either on the given path or in the class path.
* load(path): which load a list of metric values from a csv, either from the given path or from the class path.
* \__call\__(path): which given an image path will calculate its metrics and store them

Each metric contains a __Preprocess__ object that preprocess the image in order to make it usable. If none is given at creation of the metric, a preprocess with default parameters will be created (should not interfere with the image).
The parameters of the Preprocess object are stored along with the metric values as those might have an impact on the values obtained.

Use case of GaborMetrics with calculation from a csv:

    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)  #creation of a preprocess object which returns a gray level image adapted from Darter visual system
    metric = GaborMetrics(angles=[0,45,90,135], frequencies=[0.2, 0.4, 0.8],  #creation of the metric which takes parameters proper to itself,
                          preprocess=preprocess,                              #a preprocess object
                          path=os.path.join(DIR_RESULTS, CSV_GABOR))          #a saving/loading path  
    metric.metric_from_csv("Results/image_list.csv")                       #execute the metrics from a csv containing image path
    metric.save()                                                          #save the dataframe as a csv at path os.path.join(DIR_RESULTS, CSV_GABOR)
   
### Command line
This script can be used from the command line. To print the list of arguments you can type:
```
$ python Scripts/MotherMetric.py -h
```
Most of the command will take as input the path of a csv file, this csv file can be obtained with the command "list":
```
$ python Scripts/MotherMetric.py <input_path> <output_csv> list -h
```
    
## Scripts/AELauncher

This script contains the scripts used to train and test the Autoencoders in Scripts/AutoEncoders/Models.
it can be called from command line:
```
$ python Scripts/AELauncher.py -h
```
Multiple arguments can then be given to customize the training (change models, learning rate, callbacks, etc).
    
## Scripts/ClusterHabitats
 
This script contains functions used to clusterize a dataset through deep features.
```
$ python Scripts/ClusterHabitats.py -h
```
to be called it need the path of a numpy dataset and an output directory
    
## Scripts/ConvertData

This script is used to convert directories of images into another type, apply tranformations and deal with some data augmentation. 
```
$ python Scripts/ClusterHabitats.py -h
```
the positional arguments are required and are described in the help manual
    
### Scripts/test____.py
Those scripts are still in development, they are unit tests used to ensure that their corresponding program act as expected.
