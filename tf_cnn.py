# modified from saeedaghabozorgi

import time
# import tensorflow as tf

#if tensorflow 2.0 is installed
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

#import usr/local/lib/python3.7/site-packages/tensorflow_core/examples/tutorials/mnist/input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import tensorflow_datasets
# mnist = tensorflow_datasets.load('mnist')
# mnist = tensorflow_datasets.read_data_sets('mnist', one_hot=True)


import importlib.util
spec = importlib.util.spec_from_file_location("tensorflow.examples.tutorials.mnist", "/usr/local/lib/python3.7/site-packages/tensorflow_core/examples/tutorials/mnist/input_data.py")
mnist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist)
# mnist.MyClass()
# mnist['train']= '512'

tf.disable_eager_execution()

sess = tf.InteractiveSession()

x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

x_image = tf.reshape(x, [-1,28,28,1])  
x_image

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

h_conv1 = tf.nn.relu(convolve1)

conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv1

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

h_conv2 = tf.nn.relu(convolve2)

conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
conv2

layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(fcl)
h_fc1

keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN= tf.nn.softmax(fc)
y_CNN

import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.clog(y_CNN), reduction_indices=[1]))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), [1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(5000):
    setattr(mnist, 'i', 'train') #this calls '_Datasets = collections.namedtuple' in input_data.py
    start = time.time()
    batch = mnist.train.next_batch(512)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    end = time.time()
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print("step", str(i), ", training accuracy", "{:.3f}".format(train_accuracy),"test accuracy", "{:.3f}".format(test_accuracy),", B_time=" , "{:.3f}".format(end - start) )
        
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

input("Press Enter to continue...")

# !wget --output-document utils1.py http://deeplearning.net/tutorial/code/utils.py #only supported in Jupyter Notebook

import utils1
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
plt.savefig('CNN.png')
imgplot.set_cmap('gray') 

import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")
plt.savefig('CNN1.png')

ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig('CNN2.png')
    
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.savefig('CNN3.png')

    
    
sess.close() #finish the session
