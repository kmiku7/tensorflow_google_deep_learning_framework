import tensorflow as tf
import posixpath as path

data_path = path.join(path.dirname(__file__), 'data')

# demos of loading data
files = tf.train.match_filenames_once(data_path + '/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(
    files,
    shuffle=False,
    # num_epochs=1,
)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    },
)


def load_data():
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        print(sess.run(files))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(2):
            print(sess.run([features['i'], features['j']]))

        coord.request_stop()
        coord.join(threads)


# load_data()


# demos of using batch and shuffle_batch.
def demo_batch():
    example, label = features['i'], features['j']
    batch_size = 3
    capacity = 1000 + 3 * batch_size

    # batch
    # example_batch, label_batch = tf.train.batch(
    #     [example, label], batch_size=batch_size, capacity=capacity
    # )

    # shuffle batch
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=30,
    )

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        tf.local_variables_initializer().run()
        print(sess.run(files))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(40):
            cur_example_batch, cur_label_batch = sess.run(
                [example_batch, label_batch], )
            print(cur_example_batch, cur_label_batch)
        coord.request_stop()
        coord.join(threads)


demo_batch()
