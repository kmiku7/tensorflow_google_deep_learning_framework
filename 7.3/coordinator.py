import tensorflow as tf
import numpy as np
import threading
import time


def my_loop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stopping from id: %d" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d" % worker_id)
            time.sleep(1)


coord = tf.train.Coordinator()
threads = [
    threading.Thread(
        target=my_loop,
        args=(
            coord,
            i,
        ),
    ) for i in range(5)
]

for t in threads:
    t.start()

coord.join(threads)
