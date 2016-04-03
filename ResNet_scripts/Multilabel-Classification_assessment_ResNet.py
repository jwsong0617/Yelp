import numpy as np
import pandas as pd

data_root = 'F:/Yelp_data/ResNet/'

train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_photo_to_biz = pd.read_csv(data_root+'train_photo_to_biz_ids.csv', index_col='photo_id')

train_df = pd.read_csv(data_root+"train_biz_fc1000features.csv")
#test_df  = pd.read_csv(data_root+"test_biz_fc1000features.csv")

y_train = train_df['label'].values
X_train = train_df['feature vector'].values
#X_test = test_df['feature vector'].values

def convert_label_to_array(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def convert_feature_to_vector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

y_train = np.array([convert_label_to_array(y) for y in train_df['label']])
X_train = np.array([convert_feature_to_vector(x) for x in train_df['feature vector']])
#X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])


print "X_train: ", X_train.shape
print "y_train: ", y_train.shape
#print "X_test: ", X_test.shape
print "train_df:"
train_df[0:5]

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import time
t=time.time()

mlb = MultiLabelBinarizer()
y_ptrain= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix

random_state = np.random.RandomState(0)
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(X_train, y_ptrain, test_size=.2,random_state=random_state)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)) #F1 score:  0.803711220644
#classifier = OneVsOneClassifier(svm.SVC(kernel='linear', probability=True))
#classifier = OutputCodeClassifier(svm.SVC(kernel='linear', probability=True))
classifier.fit(X_ptrain, y_ptrain)

y_ppredict = classifier.predict(X_ptest)

print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

print "Samples of predicted labels (in binary matrix):\n", y_ppredict[0:3]
print "\nSamples of predicted labels:\n", mlb.inverse_transform(y_ppredict[0:3])


statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(y_ppredict, axis=0), len(y_ppredict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_ppredict)
statistics

from sklearn.metrics import f1_score

print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)