### Check the file content
import h5py
h5_root = '/mnt/'
data_root = '/home/ubuntu/kaggle/raw/'
"""
f = h5py.File(h5_root+'train_image_fc1000features.h5','r')
print 'train_image_features.h5:'
for key in f.keys():
    print key, f[key].shape
print "\nA photo:", f['photo_id'][0]
print "Its feature vector (first 10-dim): ", f['feature'][0][0:10], " ..."
f.close()
"""
### Check the file content
f = h5py.File(data_root+'test_image_fc1000features.h5','r')
for key in f.keys():
    print key, f[key].shape
print "\nA photo:", f['photo_id'][0]
print "feature vector: (first 10-dim)", f['feature'][0][0:10], " ..."
f.close()
