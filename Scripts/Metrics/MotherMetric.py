import pandas as pd

from config import *
from Preprocess import *


class MotherMetric:
    '''
    Parent class for each metric, takes care of the data processing
    '''
    def __init__(self, preprocess=None):
        if preprocess:
            self.preprocess = preprocess
        else:
            self.preprocess = Preprocess()
        self.data = pd.DataFrame()

    def function(self, image):
        '''
        perform the metric on the image 
        return the value consisting of a df with the parameters and result of the metric
        '''  
        raise NotImplementedError("Hey, Don't forget to implement the function!")

    def __call__(self, image_path=None):
        '''
        prepare the image and call the function on it, then store the result
        '''
        if not image_path:
            image = self.preprocess.get_image()
        else:
            image = self.preprocess(image_path)
        value = self.function(image)
        self.data = self.data.append(value, ignore_index=True)
        return value

    def load(data_path):
        '''
        load a csv_file
        '''
        self.data = pd.load_csv(data_path)

    def save(output_path):
        '''
        save the data as csv
        '''
        self.data.to_csv(output_path)