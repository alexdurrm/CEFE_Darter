#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code used load all images, calculate their Fourier power specturm, and save the
data in a format which is easy to run statistics with. Folders locations and
file structure are hard coded, and must be changed for different file structures
'''

import numpy as np
import imageio
import os
import csv
import cv2
import matplotlib.pyplot as plt

from pspec import get_pspec

# Resizing parameters
hab_size = (600, 900)
hab_sample = 200
fish_size = (200, 200)
fish_sample = 200

# Sampling parameters
hab_range = (10, 110)
fish_range = (10, 110)
n_samples = 2

__author__ = 'Samuel Hulse'
__email__ = 'hsamuel1@umbc.edu'

habitats = {}
habitats['barrenense'] = 'bedrock'
habitats['blennioides'] = 'boulder'
habitats['caeruleum'] = 'gravel'
habitats['camurum'] = 'boulder'
habitats['chlorosomum'] = 'sand'
habitats['gracile'] = 'detritus'
habitats['olmstedi'] = 'sand'
habitats['pyrrhogaster'] = 'sand'
habitats['swaini'] = 'detritus'
habitats['zonale'] = 'gravel'

def random_sample(data, sample_dim):
    sample_domain = np.subtract(data.shape[0:2], (sample_dim, sample_dim))       
    if sample_domain[0] == 0: sample_x = 0
    else: sample_x = np.random.randint(0, sample_domain[0])
    if sample_domain[1] == 0: sample_y = 0
    else: sample_y = np.random.randint(0, sample_domain[1])

    sample = data[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

    return sample

slopes = []
species = []
habitat = []
sexes = []
animal = []
sites = []

animals = {}
animals['barrenense'] = 'Etheostoma_barrenense_A'
animals['blennioides'] = 'Etheostoma_blennioides_A'
animals['caeruleum'] = 'Etheostoma_caeruleum_A'
animals['camurum'] = 'Nothonotus_camurus_A'
animals['chlorosomum'] = 'Etheostoma_chlorosoma_A'
animals['gracile'] = 'Etheostoma_gracile_A'
animals['olmstedi'] = 'Etheostoma_olmstedi_C'
animals['pyrrhogaster'] = 'Etheostoma_pyrrhogaster_A'
animals['swaini'] = 'Etheostoma_swaini_A'
animals['zonale'] = 'Etheostoma_zonale_A'

#Darter image processing
path = '../../Images/crops'
folders = os.listdir(path)

for folder in folders:
    current_path = os.path.join(path, folder)
    files = os.listdir(current_path)

    for image in files:
        if image.endswith('.tif'):
            img_path = os.path.join(current_path, image)
            img = imageio.imread(img_path)

            for i in range(n_samples):
                sample_dim = fish_sample
                sample_domain = np.subtract(img.shape[0:2], (sample_dim, sample_dim))       
                if sample_domain[0] == 0: sample_x = 0
                else: sample_x = np.random.randint(0, sample_domain[0])
                if sample_domain[1] == 0: sample_y = 0
                else: sample_y = np.random.randint(0, sample_domain[1])
                sample = img[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]
                sample = cv2.resize(sample, dsize=fish_size, interpolation=cv2.INTER_CUBIC)
                slope = get_pspec(sample, bin_range=fish_range)

                slopes.append(slope)
                species.append(folder)
                habitat.append(habitats[folder])
                sexes.append(image[-7])
                animal.append(animals[folder])
                sites.append(image[-12:-8])
    
with open('../fish.csv', mode='w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['species', 'animal', 'habitat', 'sex', 'slope', 'site'])
    for i in range(len(slopes)):
        writer.writerow([species[i],
            animal[i], 
            habitat[i],
            sexes[i],
            slopes[i],
            sites[i]])
