import tensorflow as tf
tf.set_random_seed(777)     # for reproducibility

# Try to find value for W and b to compute y_data = x_data * W = b
# We know that W should be 1 and b should be 0
# But let's TensorFlow figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Now we can use X and Y in place of x_data an y_data
# placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Hypothesis XW + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# launch the graph in a session
sess = tf.Session()

# initialize global variables in the graph.
sess.run(tf.global_variables_initializer())

# fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)
