data_root = 'F:/Yelp_data/caffenet/'

import numpy as np
import pandas as pd
import h5py
import time
import re #module that used to remove all characters, puntuations except digits.

test_photo_to_biz = pd.read_csv(data_root+'test_photo_to_biz.csv')
biz_ids = test_photo_to_biz['business_id'].unique()

## Load image features
f = h5py.File(data_root+'test_image_fc7features.h5','r')
image_filenames = list(np.copy(f['photo_id']))
image_filenames = [name.split('/')[-1] for name in image_filenames]  #remove the full path and the str ".jpg"
image_features = np.copy(f['feature'])
f.close()

#photo id's names don't have consistency, so i removed all characters, punctuation(.jpg) except digits
#Poor algorithm, it should be fixed later
##image_filenames preprocessing
i = 0
for x in image_filenames:
    image_filenames[i] = re.sub( "\D","",x) # remove all characters, punctuation(.jpg) except digits
    i = i+1
print "Number of business: ", len(biz_ids)

df = pd.DataFrame(columns=['business','feature vector'])
index = 0
t = time.time()

for biz in biz_ids:

    image_ids = test_photo_to_biz[test_photo_to_biz['business_id']==biz]['photo_id'].tolist()
    image_index = [image_filenames.index(str(x)) for x in image_ids]

    folder = data_root+'test_photo_folders/'
    features = image_features[image_index]
    mean_feature =list(np.mean(features,axis=0))

    df.loc[index] = [biz, mean_feature]
    index+=1
    if index%1000==0:
        print "Buisness processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

with open(data_root+"test_biz_fc7features.csv",'w') as f:
    df.to_csv(f, index=False)