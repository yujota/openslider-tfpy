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

import tensorflow as tf

import openslidertfpy.reader as reader


FILE_PATH = "/home/yota/DataForML/svs/aiba.svs"
assert os.path.isfile(FILE_PATH)


with tf.Graph().as_default():
    coord = tf.train.Coordinator()
    runner = reader.MicroImageRunner(FILE_PATH, coord)
    images, locs, levs = runner.get_inputs()  # Define images, x and y coordinates

    results = list()

    with tf.Session() as sess:
        # Some initialize function
        tf.train.start_queue_runners(sess)
        runner.start_thread(sess, [((0, 0), 2)])

        while not coord.should_stop():
            imgs, ls, vs = sess.run([images, locs, levs])
            results.extend([
                i for i, lo, lv in zip(imgs, ls, vs)
                if not reader.is_mock(lo, lv)
            ])
    print(len(results))
