import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
import time

t = time.time()

data_root = 'F:/Yelp_data/caffenet/'
data_root_ResNet = 'F:/Yelp_data/ResNet/'

train_df = pd.read_csv(data_root+"train_biz_fc7features.csv")
#test_df  = pd.read_csv(data_root+"test_biz_fc7features.csv")
train_df_Res = pd.read_csv(data_root_ResNet+"train_biz_fc1000features.csv")
#test_df_Res = pd.read_csv(data_root_ResNet+"test_biz_fc1000features.csv")

y_train = train_df['label'].values
X_train = train_df['feature vector'].values
#X_test = test_df['feature vector'].values
#y_train_Res = train_df_Res['label'].values
X_train_Res = train_df_Res['feature vector'].values
#X_test_Res = test_df_Res['feature vector'].values

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
#y_train_Res = np.array([convert_label_to_array(y) for y in train_df_Res['label']])
X_train_Res = np.array([convert_feature_to_vector(x) for x in train_df_Res['feature vector']])
#X_test_Res = np.array([convert_feature_to_vector(x) for x in test_df_Res['feature vector']])

X_train_scaled = preprocessing.scale(X_train,axis=1)
X_train_scaled_Res = preprocessing.scale(X_train_Res,axis=1)
X_train_scaled_Concat = np.hstack((X_train_scaled,X_train_scaled_Res))
#X_test_scaled = preprocessing.scale(X_test)
#X_test_scaled_Res = preprocessing.scale(X_test_Res)
#X_test_scaled_Concat = np.hstack((X_test,X_test_Res))

mlb = MultiLabelBinarizer()
y_ptrain= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix

random_state = np.random.RandomState(0)
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(X_train_scaled_Concat, y_ptrain, test_size=.2,random_state=random_state)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
classifier.fit(X_ptrain, y_ptrain)

y_ppredict = classifier.predict(X_ptest)

#print list(mlb.classes_)
#y_predict_label = mlb.inverse_transform(y_predict) #Convert binary matrix back to labels

print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"
print "Samples of predicted labels (in binary matrix):\n", y_ppredict[0:3]
print "\nSamples of predicted labels:\n", mlb.inverse_transform(y_ppredict[0:3])

statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(y_ppredict, axis=0), len(y_ppredict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_ppredict)
print statistics

print "F1 score: ", f1_score(y_ptest, y_ppredict, average='micro')
print "Individual Class F1 score: ", f1_score(y_ptest, y_ppredict, average=None)