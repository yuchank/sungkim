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
