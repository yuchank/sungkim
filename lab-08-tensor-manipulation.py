# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint

tf.set_random_seed(777)     # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# simple array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)   # rank  1
print(t.shape)  # (7,)
print(t[0], t[1], t[-1])    # 0.0, 1.0, 6.0
print(t[2:5], t[4:-1])      # [2. 3. 4.] [4. 5.]
print(t[:2], t[3:])         # [0. 1.] [3. 4. 5. 6.]

# 2D array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)       # 2
print(t.shape)      # (4, 3)

# shape, rank, axis
t = tf.constant([1, 2, 3, 4])
tf.shape(t).eval()  # array([4])

t = tf.constant([[1, 2],
                 [3, 4]])
tf.shape(t).eval()  # array([2, 2])

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()  # array([1, 2, 3, 4])

[
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        [
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]
        ]
    ]
]

# matmul vs. multiply
matrix1 = tf.constant([[3., 3.]])       # 1 x 2
matrix2 = tf.constant([[2.], [2.]])     # 2 x 1

tf.matmul(matrix1, matrix2).eval()      # 12
(matrix1 * matrix2).eval()              # matrix

# watch out broadcasting
matrix1 = tf.constant([[3., 3.]])       # 1 x 2
matrix2 = tf.constant([[2.], [2.]])     # 2 x 1
(matrix1 + matrix2).eval()      # matrix

matrix1 = tf.constant([[1., 2.]])       # 1 X 2
matrix2 = tf.constant(3.)               # const
(matrix1 + matrix2).eval()      # matrix

matrix1 = tf.constant([[1., 2.]])       # 1 X 2
matrix2 = tf.constant([3., 4.])         # list
(matrix1 + matrix2).eval()      # matrix

matrix1 = tf.constant([[1., 2.]])       # 1 X 2
matrix2 = tf.constant([[3.], [4.]])     # 2 x 1
(matrix1 + matrix2).eval()      # matrix

matrix1 = tf.constant([[3., 3.]])     # 1 X 2
matrix2 = tf.constant([[2., 2.]])     # 1 x 2
(matrix1 + matrix2).eval()      # same size

# random values for variable initializations
tf.random_normal([3]).eval()
tf.random_uniform([2]).eval()
tf.random_uniform([2, 3]).eval()

# reduce mean/sum
tf.reduce_mean([1, 2], axis=0).eval()   # integer mean 1

x = [[1., 2.],
     [3., 4.]]  # 2 x 2
tf.reduce_mean(x).eval()    # 2.5
tf.reduce_mean(x, axis=0).eval()    # [2., 3.]
tf.reduce_mean(x, axis=1).eval()    # [1.5., 3.5]
tf.reduce_mean(x, axis=-1).eval()    # [1.5., 3.5]

x = [[1., 2.],
     [3., 4.]]  # 2 x 2
tf.reduce_sum(x).eval()    # 10.0
tf.reduce_sum(x, axis=0).eval()    # [4., 6.]
tf.reduce_sum(x, axis=-1).eval()    # [3., 7.]
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()    # 5.0

# argmax with axis
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval()     # which is biggest 2, 1, 2
tf.argmax(x, axis=1).eval()     # which is biggest 2, 0
tf.argmax(x, axis=-1).eval()     # which is biggest 2, 0

# reshape, squeeze, expand_dims
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])   # 2 x 2 x 3
print(t.shape)

tf.reshape(t, shape=[-1, 3]).eval()     # -1 means i don't care. rank 2
tf.reshape(t, shape=[-1, 1, 3]).eval()     # -1 means i don't care, rank 3

tf.squeeze([[0], [1], [2]]).eval()
tf.expand_dims([0, 1, 2], 1).eval()     # list to vector

# one hot
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
t = tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
tf.reshape(t, shape=[-1, 3]).eval()

# casting
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

# stack
x = [1, 4]
y = [2, 5]
z = [3, 6]

# pack along first dim.
tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=1).eval()

# ones like and zeros like
x = [[0, 1, 2],
     [2, 1, 0]]
tf.ones_like(x).eval()
tf.zeros_like(x).eval()

# zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)

# transpose
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)  # 2 x 2 x 3
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])     # 2 x 2 x 3
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t = tf.transpose(t1, [1, 0, 2])     # 2 x 2 x 3
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))

t2 = tf.transpose(t, [1, 2, 0])     # 2 x 3 x 2
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])     # 2 x 2 x 3
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))
