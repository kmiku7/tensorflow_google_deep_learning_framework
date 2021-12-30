import tensorflow as tf
import posixpath as path

from tensorflow.python.framework.errors_impl import OutOfRangeError

input_files = [
    __file__,
    path.join(path.dirname(__file__), 'simple_demo.py'),
]
dataset = tf.data.TextLineDataset(input_files)

iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(x))
    except OutOfRangeError:
        print('finish')
