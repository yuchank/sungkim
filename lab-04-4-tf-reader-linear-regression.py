# Queue Runners
# lab 4 multi-variable linear regression
# https://www.tensorflow.org/programmers_guide/reading_data
import tensorflow as tf
tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# default values, in case of empty columns.
# Also specifies the type of the decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

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

# start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)

coord.request_stop()
coord.join(threads)

# ask my score
print('Your score will be ', sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print('Other score will be ', sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

# min_after_dequeue defines how big a buffer we will randomly sample from
#   bigger means better shuffling but slower start up and more memory used.
# capacity must be larger than min_after_dequeue and the amount larger
#   determines the maximum we will prefetch.
#   recommendation: min_after_dequeue + (num_threads + a small safety margin ) * batch_size
# min_after_dequeue = 10000
# capacity = min_after_dequeue + 3 * batch_size
# example_batch, label_batch = tf.train.shuffle_batch(
#     [example, label], batch_size=batch_size, capacity=capacity,
#     min_after_dequeue=min_after_dequeue
# )
