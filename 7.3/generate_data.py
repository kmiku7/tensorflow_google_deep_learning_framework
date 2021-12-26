import posixpath as path
import tensorflow as tf
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shards = 6
instances_per_shard = 6
data_path = path.join(path.dirname(__file__), 'data')
try:
    os.makedirs(data_path)
except FileExistsError:
    pass

for i in range(num_shards):
    filename = f'{data_path}/data.tfrecords-{i:05d}-of-{num_shards:05d}'
    print(f'filename: {filename}')
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i':
            _int64_feature(i),
            'j':
            _int64_feature(j),
        }, ), )
        writer.write(example.SerializeToString())
    writer.close()
