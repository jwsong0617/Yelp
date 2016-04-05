import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
import time

t = time.time()

data_root = 'F:/Yelp_data/caffenet/'
data_root_ResNet = 'F:/Yelp_data/ResNet/'

train_df = pd.read_csv(data_root+"train_biz_fc7features.csv")
test_df  = pd.read_csv(data_root+"test_biz_fc7features.csv")
train_df_fc6 = pd.read_csv(data_root+"train_biz_fc6features_batch500.csv")
test_df_fc6  = pd.read_csv(data_root+"test_biz_fc6features_batch500.csv")
train_df_Res = pd.read_csv(data_root_ResNet+"train_biz_fc1000features.csv")
test_df_Res = pd.read_csv(data_root_ResNet+"test_biz_fc1000features.csv")

y_train = train_df['label'].values
X_train = train_df['feature vector'].values
X_test = test_df['feature vector'].values
#y_train_fc6 = train_df['label'].values
X_train_fc6 = train_df_fc6['feature vector'].values
X_test_fc6 = test_df_fc6['feature vector'].values
#y_train_Res = train_df_Res['label'].values
X_train_Res = train_df_Res['feature vector'].values
X_test_Res = test_df_Res['feature vector'].values

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
X_train_fc6 = np.array([convert_feature_to_vector(x) for x in train_df_fc6['feature vector']])
X_test_fc6 = np.array([convert_feature_to_vector(x) for x in test_df_fc6['feature vector']])
#y_train_Res = np.array([convert_label_to_array(y) for y in train_df_Res['label']])
X_train_Res = np.array([convert_feature_to_vector(x) for x in train_df_Res['feature vector']])
X_test_Res = np.array([convert_feature_to_vector(x) for x in test_df_Res['feature vector']])

X_train_scaled = preprocessing.normalize(X_train, norm='l2')
X_train_scaled_Res = preprocessing.normalize(X_train_Res, norm='l2')
X_train_scaled_fc6 = preprocessing.normalize(X_train_fc6, norm='l2')
X_train_scaled_Concat = np.hstack((X_train_scaled,X_train_scaled_Res,X_train_scaled_fc6))
X_test_scaled = preprocessing.normalize(X_test, norm='l2')
X_test_scaled_Res = preprocessing.normalize(X_test_Res, norm='l2')
X_test_scaled_fc6 = preprocessing.normalize(X_test_fc6, norm='l2')
X_test_scaled_Concat = np.hstack((X_test_scaled,X_test_scaled_Res,X_test_scaled_fc6))

mlb = MultiLabelBinarizer()
y_train= mlb.fit_transform(y_train)  #Convert list of labels to binary matrix

random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
classifier.fit(X_train_scaled_Concat, y_train)

y_predict = classifier.predict(X_test_scaled_Concat)

#print list(mlb.classes_)
y_predict_label = mlb.inverse_transform(y_predict) #Convert binary matrix back to labels

print "Time passed: ", "{0:.1f}".format(time.time()-t), "sec"

test_data_frame  = pd.read_csv(data_root+"test_biz_fc7features.csv") #fc7features and fc1000features have same business names
df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(test_data_frame)):
    biz = test_data_frame.loc[i]['business']
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(data_root+"submission_fc6_batch500_fc7_fc1000_norm.csv",'w') as f:
    df.to_csv(f, index=False)

statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
statistics.loc["biz count"] = np.append(np.sum(y_predict, axis=0), len(y_predict))
pd.options.display.float_format = '{:.0f}%'.format
statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_predict)
print statistics