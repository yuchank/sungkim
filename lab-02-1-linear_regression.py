import tensorflow as tf
tf.set_random_seed(777)     # for reproducibility

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Try to find value for W and b to compute y_data = x_data * W = b
# We know that W should be 1 and b should be 0
# But let's TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis XW + b
hypothesis = x_train * W + b

# cost/loss function 
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# launch the graph in a session
sess = tf.Session()

# initialize global variables in the graph.
sess.run(tf.global_variables_initializer())

# fit the line 
for step in range(2001):
    sess.run(train)
    if step % 200 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
