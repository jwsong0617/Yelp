#Uncomment if skip previous train
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import time

t = time.time()

data_root = 'F:/Yelp_data/caffenet/'
train_df = pd.read_csv(data_root+"train_biz_fc7features.csv")
test_df  = pd.read_csv(data_root+"test_biz_fc7features.csv")

y_train = train_df['label'].values
X_train = train_df['feature vector'].values
X_test = test_df['feature vector'].values

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
X_test = np.array([convert_feature_to_vector(x) for x in test_df['feature vector']])

mlb = MultiLabelBinarizer()
y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix

random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

#print list(mlb.classes_)
y_predict_label = mlb.inverse_transform(y_predict) #Convert binary matrix back to labels

print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

test_data_frame  = pd.read_csv(data_root+"test_biz_fc7features.csv")
df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(test_data_frame)):
    biz = test_data_frame.loc[i]['business']
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submission_fc7.csv",'w') as f:
    df.to_csv(f, index=False)

statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(y_predict, axis=0), len(y_predict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_predict)
print statistics