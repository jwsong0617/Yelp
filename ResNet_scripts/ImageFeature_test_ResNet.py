caffe_root = '/home/ubuntu/caffe/'
data_root = '/home/ubuntu/kaggle/raw/'
h5_root = '/mnt/'

import numpy as np
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

import os
"""
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    !caffe_root/scripts/download_model_binary.py ../models/bvlc_reference_caffenet
"""
## Use GPU
caffe.set_device(0)
caffe.set_mode_gpu()
#These sector is modified for use ResNet as pre-trained model from Kaggle Forum Deep learning starter code which use Caffenet as pretrained model
def extract_features(images, layer = 'fc1000'):
    net = caffe.Net(caffe_root + 'models/ResNet/ResNet-50-deploy.prototxt',
                caffe_root + 'models/ResNet/ResNet-50-model.caffemodel',
                caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  #same sequence as caffenet(RGB -> BGR)
    #mean pixel is changed ilsvrc... to ResNet_mean.npy which is generated from ResNet_mean.binaryphoto
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ResNet_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1] / same as caffenet
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]] as same as caffenet

    num_images= len(images)
    net.blobs['data'].reshape(num_images,3,224,224) # changed 3,227,227 to 3, 224,224. Each numeric fields mean number of photo, channel, height, width
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data

import h5py
#f.close()
f = h5py.File(data_root+'test_image_fc1000features.h5','w')
filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = f.create_dataset('feature',(0,1000), maxshape = (None,1000)) # 4096 to 1000
f.close()

import pandas as pd
test_photos = pd.read_csv(data_root+'test_photo_to_biz_ids.csv')
test_folder = data_root+'test_photos/test_photos/'
test_images = [os.path.join(test_folder, str(x)+'.jpg') for x in test_photos['photo_id'].unique()]  # get full filename

num_test = len(test_images)
print "Number of test images: ", num_test
#batch size 500,256,200,128,64,32 are out of memory
batch_size = 16

# Test Images
for i in range(0, num_test, batch_size):
    images = test_images[i: min(i+batch_size, num_test)]
    features = extract_features(images, layer='fc1000')
    num_done = i+features.shape[0]
    f= h5py.File(data_root+'test_image_fc1000features.h5','r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done,features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done%20000==0 or num_done==num_test:
        print "Test images processed: ", num_done