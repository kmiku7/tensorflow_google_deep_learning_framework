import tensorflow as tf

# q = tf.FIFOQueue(2, "int32")
q = tf.RandomShuffleQueue(2, dtypes='int32', min_after_dequeue=0)
init = q.enqueue_many(([0,10],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)
print(q)