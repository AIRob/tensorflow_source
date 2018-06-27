# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:34:12 2018

@author: lWX379138
"""
import numpy as np
from scipy.misc import imsave

filename = 'cifar-10-binary/cifar-10-batches-bin/test_batch.bin'

bytestream = open(filename,'rb')
buf = bytestream.read(10000*(1+32*32*3))
bytestream.close()

data = np.frombuffer(buf,dtype=np.uint8)
data = data.reshape(10000,1+32*32*3)
labels_images = np.hsplit(data,[1])
labels = labels_images[0].reshape(10000)
images = labels_images[1].reshape(10000,32,32,3)

img = np.reshape(images[0],(3,32,32))
img = img.transpose(1,2,0)

import pylab

print(labels[0])
pylab.imshow(img)
pylab.show()
