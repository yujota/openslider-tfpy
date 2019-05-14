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
import numpy as np
from PIL import Image

from openslidertfpy import MicroPatchReader, is_mock


FILE_PATH = "/path/to/wsi/file.svs"
assert os.path.isfile(FILE_PATH)


with tf.Graph().as_default():
    read_patch_coordinator = tf.train.Coordinator()
    runner = MicroPatchReader(
        FILE_PATH, read_patch_coordinator, image_width=500, image_height=500
    )
    images, locs, levs = runner.get_inputs()

    results = list()

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        runner.start_thread([((0, 0), 2)])

        while not read_patch_coordinator.should_stop():
            imgs, ls, vs = sess.run([images, locs, levs])
            results.extend([
                i for i, lo, lv in zip(imgs, ls, vs)
                if not is_mock(lo, lv)
            ])

    # Show the result
    loaded_image_array = results[0]
    loaded_image = Image.fromarray(loaded_image_array.astype(np.uint8))
    loaded_image.show()
