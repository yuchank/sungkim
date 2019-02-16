# Queue Runners
# lab 4 multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
import tensorflow as tf
tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer()


# make sure the shape and data are OK
print(x_data, '\nx_data shape:', x_data.shape)
print(y_data, '\ny_data shape:', y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# X(m 3) x W(3, 1) + b(element-wise)
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the group
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)

# ask my score
print('Your score will be ', sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print('Other score will be ', sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
