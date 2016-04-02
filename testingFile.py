data_root = 'F:/Yelp_data/caffenet/'

import numpy as np
import pandas as pd
import h5py
import re
import time

test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
biz_ids = test_photo_to_biz['business_id'].unique()

## Load image features
f = h5py.File(data_root+'test_image_fc7features.h5','r')
image_filenames = list(np.copy(f['photo_id']))
image_filenames = [name.split('/')[-1] for name in image_filenames]  #remove the full path and the str ".jpg"
image_features = np.copy(f['feature'])
f.close()
#print "\nA photo:", image_filenames[10:50]
#print re.sub( "\D","",image_filenames[1])
#print image_filenames[20:30]
i = 0
for x in image_filenames:
    image_filenames[i] = re.sub( "\D","",x)
    i = i+1
print image_filenames[20:50]