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
"""TensorFlow Custom runner to produce micro images from pathology file

Usage:

import tensorflow as tf


openslide_read_region_params = [
    ((100, 100), 0),  # location and level for openslide's read_region func
    ((200, 200), 0),
    ((300, 300), 1)
]
image_width, image_height = 128, 128

with tf.Graph().as_default():
    coordinator = tf.train.Coordinator()
    runner = MicroImageReader(
        "sample.svs", coordinator, image_width, image_height
    )
    images, locations_batch, levels_batch = runner.get_inputs()
    # Define images, openslide's read_region function parameters

    results = some_op(images)

    with tf.Session() as sess:

        # Some function to initialize values should be placed here

        # Start reading pathology images
        tf.train.start_queue_runners(sess)
        runner.start_thread(sess)
        while not coord.should_stop():
            actual_results, locs, levs = sess.run(
                [results, locations_batch, levels_batch]
            )
"""
import time
import threading
import typing
import queue

import openslide
import numpy as np
import tensorflow as tf


if typing.TYPE_CHECKING:
    import tensorflow.python.training.coordinator
    import tensorflow.python.client.session
    import tensorflow.python.framework.ops
    TfCoordinator = tensorflow.python.training.coordinator.Coordinator
    TfSession = tensorflow.python.client.session.Session
    ReadRegionLocation = typing.Tuple[int, int]
    ReadRegionLevel = int
    ReadRegionParam = typing.Tuple[ReadRegionLocation, ReadRegionLevel]
    ImageBatchPlaceholder = tensorflow.python.framework.ops.Tensor
    LocationBatchPlaceholder = tensorflow.python.framework.ops.Tensor
    LevelBatchPlaceholder = tensorflow.python.framework.ops.Tensor


class MicroImageReader(object):

    def __init__(
            self,
            svs_path: str,
            coordinator: 'TfCoordinator',
            image_width: int = 64,
            image_height: int = 64,
            num_worker: int = 4,
            fifo_queue_capacity: int = 20000,
            batch_size: int = 100,
            verbose: bool = False
    ):
        self.num_channel = 3
        self.image_width = image_width
        self.image_height = image_height
        self.coordinator = coordinator
        self.img_placeholder = tf.placeholder(
            dtype=tf.int64,
            shape=[
                image_height, image_width, self.num_channel
            ])
        self.location_placeholder = tf.placeholder(dtype=tf.int64, shape=[2])
        self.level_placeholder = tf.placeholder(dtype=tf.int64, shape=[])
        self.queue = tf.FIFOQueue(
            capacity=fifo_queue_capacity,
            dtypes=[tf.int64, tf.int64, tf.int64],
            shapes=[[image_height, image_width, self.num_channel], [2], []]
        )
        self.enqueue_op = self.queue.enqueue([
            self.img_placeholder,
            self.location_placeholder,
            self.level_placeholder,
        ])
        self.svs_path = svs_path
        self.num_worker = num_worker
        self.read_region_queue = queue.Queue()
        self.batch_size=batch_size
        self.verbose = verbose

    def start_thread(
            self,
            sess: 'TfSession',
            read_region_parameters: typing.Iterable['ReadRegionParam']
    ) -> None:
        """Start enqeue operation and reading file"""
        for p in read_region_parameters:
            self.read_region_queue.put(p)
        if self.verbose:
            print("Start threading")
        for i in range(self.num_worker):
            t = threading.Thread(
                target=self._thread_worker, args=(sess, )
            )
            t.start()
        t = threading.Thread(
            target=self._finish_enqeue, args=(sess, )
        )
        t.start()

    def _thread_worker(self, sess):
        slide = openslide.OpenSlide(self.svs_path)
        while not self.read_region_queue.empty():
            location, level = self.read_region_queue.get()
            tile = slide.read_region(
                location=location,
                level=level,
                size=(self.image_width, self.image_height)
            )
            img_array = np.array(tile.convert("RGB"))
            self._enqueue(sess, img_array, location, level)
            self.read_region_queue.task_done()
        slide.close()

    def _enqueue(self, sess, image, location, level):
        sess.run(self.enqueue_op, feed_dict={
            self.img_placeholder: image,
            self.location_placeholder: location,
            self.level_placeholder: level,
        })

    def _finish_enqeue(self, sess):
        self.read_region_queue.join()
        if self.verbose:
            print("Remaining...", sess.run(self.queue.size()))
        while not sess.run(self.queue.size()) == 0:
            time.sleep(1)
        self.coordinator.request_stop()
        if self.verbose:
            print("Finished threading")
        mock_img = np.zeros(
            shape=[self.image_height, self.image_width, self.num_channel]
        )
        mock_location, mock_level, mock_size = (-1, -1), -1, (-1, -1)
        try:  # Place mock images to fill batch which is fixed size
            for i in range(self.batch_size):
                self._enqueue(
                    sess, mock_img, mock_location, mock_level
                )
        except Exception as e:
            pass

    def get_inputs(self) -> typing.Tuple[
            'ImageBatchPlaceholder',
            'LocationBatchPlaceholder',
            'LevelBatchPlaceholder'
    ]:
        """Return tensor op of images, locations and levels"""
        images_batch, locations_batch, levels_batch = \
            self.queue.dequeue_many(self.batch_size)
        return images_batch, locations_batch, levels_batch


def is_mock(location: np.ndarray, level: np.ndarray) -> bool:
    return np.array_equal(location, np.array((-1, -1))) and level == -1
