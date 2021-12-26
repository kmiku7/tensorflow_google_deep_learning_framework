import tensorflow as tf

import image_preprocess

datafile_pattern = ""
files = tf.train.match_filenames_once(datafile_pattern)
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    },
)
image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

image_size = 299
distorted_image = image_preprocess.preprocess_for_train(
    decoded_image,
    image_size,
    image_size,
    None,
)
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue,
)


def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


def calc_loss(image_batch, label_batch):
    INPUT_NODE = 784
    OUTPUT_NODE = 10
    LAYER1_NODE = 500
    REGULARAZTION_RATE = 0.0001

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(image_batch, weights1, biases1, weights2, biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,
        labels=label_batch,
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    return loss


learning_rate = 0.01
# logit = inference(image_batch)  # noqa: F821
loss = calc_loss(image_batch, label_batch)  # noqa: F821
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run((
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
    ), )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
