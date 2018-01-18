import tensorflow as tf
import numpy as np

SEED = 43

sess = tf.Session()

train_data = np.expand_dims(np.load("train_data.npy"), axis=-1)
results = np.expand_dims(np.load("results.npy"), axis=-1)

train_data_node = tf.placeholder(
    np.float32,
    shape=train_data.shape)

# print(results[0])
# print(results.shape)
results_node = tf.placeholder(np.float32, shape=results.shape)

conv1_weights = tf.Variable(
    tf.truncated_normal([3, 3, 1, 1],  # 3x3 filter; 1 filter
                        stddev=0.0,
                        seed=SEED, dtype=np.float32))
# conv1_weights = tf.Variable(
#     tf.truncated_normal([3, 3, 1, 1],  # 3x3 filter; 1 filter
#                         stddev=0.1,
#                         seed=SEED, dtype=np.float32))

conv1_biases = tf.Variable(tf.zeros([1], dtype=np.float32))

# print(train_data.shape)
# print(np.expand_dims(train_data, axis=-1).shape)
# assert False

def model(data, train=False):
    conv = data + tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    return conv

loss = tf.reduce_mean(tf.square(results_node - model(train_data_node)))

learning_rate = 0.25
init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

feed_dict = {
    train_data_node: train_data,
    results_node: results
}

sess.run(init, feed_dict)


NUM_EPOCHS = 100
for i in range(NUM_EPOCHS):
    sess.run(optimizer, feed_dict)
    print(sess.run(loss, feed_dict))

print(conv1_weights.eval(sess))
