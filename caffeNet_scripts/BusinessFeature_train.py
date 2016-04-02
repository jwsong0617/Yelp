data_root = 'F:/Yelp_data/caffenet/'

import numpy as np
import pandas as pd
import h5py
import time

train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_labels = pd.read_csv(data_root+'train.csv').dropna()
train_labels['labels'] = train_labels['labels'].apply(lambda x: tuple(sorted(int(t) for t in x.split())))
train_labels.set_index('business_id', inplace=True)
biz_ids = train_labels.index.unique()
print "Number of business: ", len(biz_ids) ,   "(4 business with missing labels are dropped)"

## Load image features
f = h5py.File(data_root+'train_image_fc7features.h5','r')
train_image_features = np.copy(f['feature'])
f.close()


t= time.time()
## For each business, compute a feature vector
df = pd.DataFrame(columns=['business','label','feature vector'])
index = 0
for biz in biz_ids:

    label = train_labels.loc[biz]['labels']
    image_index = train_photo_to_biz[train_photo_to_biz['business_id']==biz].index.tolist()
    folder = data_root+'train_photo_folders/'

    features = train_image_features[image_index]
    mean_feature =list(np.mean(features,axis=0))

    df.loc[index] = [biz, label, mean_feature]
    index+=1
    if index%1000==0:
        print "Buisness processed: ", index, "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

with open(data_root+"train_biz_fc7features.csv",'w') as f:
    df.to_csv(f, index=False)