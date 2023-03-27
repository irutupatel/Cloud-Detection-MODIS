#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import os
import pickle


# In[38]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


home = '/home/hackathon/output_64_Javier_labelled/'


# In[ ]:





# In[20]:


all_files = []
all_features = []
all_classifications = []

errors = 0
no_files = 0
for file_name in os.listdir(home):
    if file_name.endswith('.hdf'):
        no_files += 1
        path = home + file_name
        try:
            file = h5py.File(path, 'r')
            
            files = []
            classifications = []
            features = []
            for key in list(file.keys()):
                accuracy = file[key + '/ClassificationAccuracy'][()]
                if accuracy == 1:
                    all_files.append(file)
                    all_features.append(file[key + '/ImageFeatures'][()])
                    all_classifications.append(file[key + '/ImageClassification'][()])
        except OSError:
            errors += 1
print(f'Successfully loaded {len(all_features)} images from {no_files - errors}/{no_files} files')


# In[61]:


classification = all_classifications
for i in range(1, len(all_classifications)):
    classification[i][classification[i] > 0] = 1
classification.remove(1)
one_array = np.ones([64, 64])
classification.insert(0, one_array)


# In[170]:


reshape_arr = np.reshape(all_features, (images * pixels, 43))
reshape_df = pd.DataFrame(reshape_arr)
reshape_df


# In[173]:


all_classifications


# In[163]:


images = len(all_features)
pixels = len(all_features[0][0]) * len(all_features[0][0][0])
array = np.empty([images * pixels, len(all_features[0])])

for i in range(len(all_features)):
    for j in range(len(all_features[i])):
        for k in range(len(all_features[i][j])):
            for l in range(len(all_features[i][j][k])):
                array[115 * i + 64 * k + l][j] = all_features[i][j][k][l]
df = pd.DataFrame(array)
df.shape


# In[165]:





# In[108]:


class_array = np.empty([images * pixels])
for i in range(len(all_classifications)):
    for j in range(len(all_classifications[i])):
        for k in range(len(all_classifications[i][j])):
            class_array[i * j * k] = int(all_classifications[i][j][k])


# In[109]:


class_array


# In[101]:


all_classifications[4][62]


# In[103]:


list_array = list(class_array)
list_array


# In[110]:


df['Classification'] = class_array


# In[111]:


ones = df[df['Classification'] == 1]
ones


# In[ ]:


path = '~/'
pd.to_csv(df)


# In[162]:


for i in range(115):
    f = plt.figure()
    plot = plt.imshow(all_features[i][26])
    cbar = f.colorbar(plot)


# In[142]:


dropped_df = df.replace(0, np.nan)
dropped_df = dropped_df.dropna(how='all', axis=0)


# In[143]:


dropped_df.shape


# In[133]:


dropped_df


# In[130]:




