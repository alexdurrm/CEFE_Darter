#setup
export PYTHONPATH="absolute/path/to/CEFE/"

you can create a new conda environment with:
conda create --name <envname> --file conda_list

# CEFE_Darter
**_Script/_** contains all the Scripts from this project.
**_Script/Metrics/_** contains the python scripts used on images that output csv files in **_Result/_** directory.

## Metrics

All python scripts ending with __"Metrics"__ contains one or multiple classes that inherits from __MotherMetric__ and that takes care of a specific type of metric.
All those classes have the following functions:
* function(image): which take an image as input and execute the metric on one image.
* metric_from_csv(csv_path): which allows to execute the metric on a list of images contained in a csv and store those metrics internally.
* save(path): which save the list of metric values as a csv, either on the given path or in the class path.
* load(path): which load a list of metric values from a csv, either from the given path or from the class path.
* visualize(): which allows to visualize the metric values handled by this class.

Each metric contains a __Preprocess__ object that preprocess the image in order to make it usable. If none is given at creation of the metric, it is created by default with preset parameters.
The parameters of the Preprocess object are stored along with the metric values as those might have an impact on the values obtained.

Use case of GaborMetrics with calculation from a csv:

    preprocess = Preprocess(img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)  #creation of a preprocess object which returns a gray level image adapted from Darter visual system
    metric = GaborMetrics(angles=[0,45,90,135], frequencies=[0.2, 0.4, 0.8],  #creation of the metric which takes parameters proper to itself,
                          preprocess=preprocess,                              #a preprocess object
                          path=os.path.join(DIR_RESULTS, CSV_GABOR))          #a saving/loading path  
    metric.metric_from_csv("Results/image_list.csv")                       #execute the metrics from a csv containing image path
    metric.save()                                                          #save the dataframe as a csv at path os.path.join(DIR_RESULTS, CSV_GABOR)
    
Use case of GaborMetrics for simple visualisation:

    metric = GaborMetrics(angles=[0,45,90,135], frequencies=[0.2, 0.4, 0.8])  #creation of the metric with default values
    metric.load("path_gabor_values.csv")                                      #load the metric values of the csv
    metric.visualize()                                                        #open a plot to visualize its internal data
   
## Command line
The main file of the project is __run_metrics.py__, it will create a csv file containing a list of image path and informations from a given repertory. It will also create a csv file called experiments.csv corresponding to each styleTransfer combinaison.
```
$ python Scripts/Metrics/run_metrics Directory -d 2
```
-d is the depth of the Directory search, 0 is for a single image, 1 is for a simple directory, 2 is for directory and subdirs.

You can also run the metrics directly from command line by feeding it the csv output of run_metric.py:
```
$ python Scripts/Metrics/GaborMetrics.py work Results/image_list.csv
```
If the command word is _work_ the metric will be calculated from the input data and saved, if the command is _visu_ the previous metrics will be loaded and visualized.
```
$ python Scripts/Metrics/GaborMetrics visu Results/image_list.csv
```
