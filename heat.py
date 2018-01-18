import tensorflow as tf
import numpy as np

#Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

import matplotlib.pyplot as plt


sess = tf.InteractiveSession()


def DisplayArray(a, fmt='jpeg', rng=[0,1]):
    """Display an array as a picture."""
    a = (a - rng[0])/float(rng[1] - rng[0])*255
    a = np.uint8(np.clip(a, 0, 255))
    plt.imshow(a)
    plt.pause(0.005)


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([
        [0.5, 1.0, 0.5],
        [1.0, -6., 1.0],
        [0.5, 1.0, 0.5]
    ])
    return simple_conv(x, laplace_k)


def exact_heat(u_init):
    eps = tf.placeholder(tf.float32, shape=())
    
    U  = tf.Variable(u_init)
    U_ = U + eps*laplace(U)
    # U_ = eps.laplace(U)
    
    step = tf.group(U.assign(U_))
    
    tf.global_variables_initializer().run()
    step.run({eps: 1.0})
    return U.eval()


# rng = np.random.RandomState(42)

train_data = np.random.rand(20, 10, 10).astype(np.float32) # num_samples, (M x N)
results = np.array([exact_heat(x) for x in train_data])

np.save("train_data.npy", train_data)
np.save("results.npy", results)
