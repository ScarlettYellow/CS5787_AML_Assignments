#!/usr/bin/env python
# coding: utf-8

# # Question 2
# ## 2-b Preprocessing the input
# 
# 1.**Unit-Normalization**: To avoid overfitting, I used Unit-Normalization, that is, mapping all coordinate locations to the [0,1] range.
# 
# 
# 2.**Standardization**: I didn't use the standardization method. Because standardization is not required for tree-based models (e.g. random forest, Bagging and boosting) which are not sensitive to variable size.
# 
# 
# 3.**Mean subtraction**: I didn't use mean subtraction, because it's not necessary to do this in this task. 
# 
# ## 2-c Preprocessing the output
# 
# 1.Rescale the pixel intensities to lie between 0.0 and 1.0.
# 
# 2.Convert the image to grayscale.

# In[1]:


# import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os
from skimage import data,filters,io,transform,feature,segmentation,restoration,util,color


# In[2]:


# use color image 

img = io.imread('data/monalisa_ori.jpg') # read image
print (img.shape)
print (img.ndim)
io.imshow(img)
print(img[:2])

# gray: default 0-1
# RGB: default 0-255
img = np.array(img, dtype=np.float32)/255.0
img


# In[3]:


# meshgrid, get coordinate locations

x = np.arange(0, img.shape[1])
y = np.arange(0, img.shape[0])
locations = np.meshgrid(y,x)
locations = np.stack(locations, axis=-1).reshape(-1, 2)

locations.shape


# In[4]:


# sample points: uniformly sample 5,000 random (x,y) coordinate locations, to build a training set

num_sample_points = 5000
np.random.shuffle(locations)
sample_points = locations[:num_sample_points]
test_points = locations[num_sample_points:]

print (sample_points[:10])


# In[5]:


# pixels match

sample_point_pixels = np.array([img[x[0], x[1]] for x in sample_points], dtype='float32')
test_point_pixels = np.array([img[x[0], x[1]] for x in test_points], dtype='float32')


# In[6]:


# normalize the coordinates

normalized_sample_points = sample_points.copy().astype('float32')
normalized_test_points = test_points.copy().astype('float32')

normalized_sample_points[:, 0] /= img.shape[0]
normalized_sample_points[:, 1] /= img.shape[1]
normalized_test_points[:, 0] /= img.shape[0]
normalized_test_points[:, 1] /= img.shape[1]


# In[7]:


print (sample_point_pixels[:5])
print (test_point_pixels[:5])


# In[8]:


print (normalized_sample_points[:5])
print (normalized_test_points[:5])


# In[9]:


x_train = normalized_sample_points
y_train = sample_point_pixels
x_test = normalized_test_points
y_test = test_point_pixels

