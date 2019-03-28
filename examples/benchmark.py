#   Copyright
#     2019 Department of Dermatology, School of Medicine, Tohoku University
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import os
import time
import functools
import random
import pickle

import tensorflow as tf

from openslidertfpy import MicroImageReader


FILE_PATH = "/home/yota/DataForML/svs/aiba.svs"  # Place WSI path here
assert os.path.isfile(FILE_PATH)


def run(read_region_params, width, height, num_worker=4, batch_size=100):
    with tf.Graph().as_default():
        coord = tf.train.Coordinator()
        runner = MicroImageReader(
            FILE_PATH,
            coord,
            image_width=width,
            image_height=height,
            num_worker=num_worker,
            batch_size=batch_size
        )
        images, _, _ = runner.get_inputs()

        with tf.Session() as sess:
            tf.train.start_queue_runners(sess)
            runner.start_thread(sess, read_region_params)

            while not coord.should_stop():
                sess.run([images])


def make_params(num_images):
    f = functools.partial(random.randrange, 0, 10000)
    return [((f(), f()), 0) for _ in range(num_images)]


def measure_benchmark(img_size=64, num_images=1000):
    w, h = img_size, img_size
    num_thread = [1, 2, 3, 4, 5, 6, 7, 8]
    ps = make_params(num_images)
    results = list()
    for n in num_thread:
        start = time.time()
        run([], w, h, n)
        offset = time.time() - start
        start = time.time()
        run(ps, w, h, n)
        duration = time.time() - start - offset
        results.append((n, duration))
        print("Th: {}, Isize {}: {}".format(n, img_size, duration))
    d = {"image_size": img_size, "num_images": num_images, "results": results}
    f_name = "benchmark_{}_{}.pickle".format(img_size, num_images)
    with open(f_name, "wb") as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    for i in [32, 64, 128, 256]:
        measure_benchmark(i, 5000)
