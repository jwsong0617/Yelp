import numpy as np
caffe_root = '/home/ubuntu/caffe/'
#np.save('./tmp/123',np.array([103.939, 116.779, 123.68]))
a = np.load(caffe_root + 'python/caffe/imagenet/ResNet_mean.npy')
b = np.load(caffe_root + 'python/caffe/imagenet/ResNet_mean.npy').mean(1).mean(1)
print 'ResNet_mean: ', a
print 'ResNet_mean_mean: ', b