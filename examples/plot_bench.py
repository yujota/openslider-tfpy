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
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


FILE_PH = "benchmark_{}_{}.pickle"
PARAMS = [(32, 5000), (64, 5000), (128, 5000), (256, 5000)]


def get_y():
    ys = list()
    for i_size, n_images in PARAMS:
        f_name = FILE_PH.format(i_size, n_images)
        with open(f_name, "rb") as f:
            d = pickle.load(f)
            durations = [r[-1] for r in d["results"]]
            durations = np.array(durations) / n_images
            ys.append(durations)
    return ys


def plot():
    xs = np.arange(1, 9)
    ys = get_y()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for y, i_size in zip(ys, (32, 64, 128, 256)):
        ax.plot(xs, y, label="{} x {}".format(i_size, i_size), marker="o")
    ax.legend(title="Image size")
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Time to read one micro pathology image [sec]")
    plt.title("Benchmarks of openslider-tfpy's MicroImageReader")
    plt.show()


if __name__ == "__main__":
    plot()
