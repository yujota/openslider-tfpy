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
"""TensorFlow Custom runner to produce pathology images"""
import time
import threading
import queue

import openslide
import numpy as np
import tensorflow as tf


class MicroImageRunner(object):

    def __init__(
            self,
            svs_path,
            coordinator,
            image_width=64,
            image_height=64,
            num_worker=4,
            fifo_queue_capacity=20000,
            batch_size=100,
            verbose=True
    ):
        """
        :param str svs_path:
        :param tensorflow.python.training.coordinator.Coordinator coordinator:
        """
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

    def start_thread(self, sess, read_region_parameters) -> None:
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
        """Helper method to enqeue image

        :param tensorflow.python.client.session.Session sess:
        """
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

    def get_inputs(self):
        """Return tensor op of images, x and y coordination"""
        images_batch, locations_batch, levels_batch = \
            self.queue.dequeue_many(self.batch_size)
        return images_batch, locations_batch, levels_batch


def is_mock(location, level):
    return np.array_equal(location, np.array((-1, -1))) and level == -1
