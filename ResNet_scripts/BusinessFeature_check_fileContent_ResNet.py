# import numpy as np
data_root = 'F:/Yelp_data/ResNet/'
import h5py
import pandas as pd
"""
f = h5py.File(data_root+'train_image_fc7features.h5','r')
print 'train_image_features.h5:'
for key in f.keys():
    print key, f[key].shape

print "\nA photo:", f['photo_id'][10200:10300]
print "Its feature vector (first 10-dim): ", f['feature'][0][0:10], " ..."
f.close()
"""
"""
f = h5py.File(data_root+'test_image_fc7features.h5','r')
for key in f.keys():
    print key, f[key].shape
print "\nA photo:", f['photo_id'][0:30]
print "feature vector: (first 10-dim)", f['feature'][0][0:10], " ..."
f.close()
"""
"""
#train business check

train_business = pd.read_csv(data_root + 'train_biz_fc1000features.csv')
print train_business.shape
print train_business[0:5]
"""
#test business check

test_business = pd.read_csv(data_root + 'test_biz_fc1000features.csv')
print test_business.shape
#test_business[0:5]
print test_business[0:5]
