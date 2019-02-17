# lab 4 multi-variable linear regression

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[0.0, 0.0],
          [0.1, 0.1],
          [0.2, 0.4],
          [0.3, 0.9],
          [0.4, 0.16],
          [0.5, 0.25],
          [0.6, 0.36],
          [0.7, 0.49],
          [0.8, 0.64],
          [0.9, 0.81],
          [1.0, 1.0]]
y_data = [[1.12],
          [1.05],
          [0.72],
          [0.49],
          [0.36],
          [0.12],
          [0.21],
          [0.01],
          [0.34],
          [0.35],
          [0.45]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# X(m, 2) x W(2, 1) + b(element-wise)
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the group
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, w_val, b_val, _ = sess.run([cost, hypothesis, W, b, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val, w_val, b_val)
