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
import re
import subprocess
from setuptools import setup, find_packages


REQUIRED_PACKAGES = [
    "openslide-python",
    'numpy',
    'tensorflow==1.4.0',
]


def write_version2initpy(version_str):
    version_py_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "openslidertfpy",
        "__init__.py"
    )
    result = list()
    with open(version_py_path, "rt") as f:
        for l in f.readlines():
            t = re.sub(
                r'__version__ = \"[\w|\.|\-|\+]+\"',
                '__version__ = "{}"'.format(version_str),
                l
            )
            result.append(t)
    with open(version_py_path, "wt") as f:
        f.writelines(result)


def get_version():
    v = subprocess.run("git tag", shell=True, stdout=subprocess.PIPE)
    versions = v.stdout.decode("utf8")
    vs = versions.splitlines()
    return vs[-1]


def main():
    try:
        version = get_version()
    except:
        version = "0.0.1-alpha"
    write_version2initpy(version)
    setup(
        name="openslidertfpy",
        version=version,
        description=
        "TensorFlow Custom runner to produce micro images from pathology file",
        author="Yuji Ota",
        url="https://github.com/OtaYuji/openslider-tfpy",
        packages=find_packages(
            exclude=["*.tests", "*.test.*", "tests.*", "tests"]
        ),
        license='Apache License 2.0',
        install_requires=REQUIRED_PACKAGES,
    )


if __name__ == "__main__":
    main()
