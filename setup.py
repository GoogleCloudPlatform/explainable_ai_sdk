# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sets up explainability SDK package."""

from os import path
import setuptools

__package_name__ = 'explainable_ai_sdk'

# Get version from version module.
with open(__package_name__ + '/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  
__version__ = globals_dict['__version__']

# Read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()
  long_description = long_description.replace(
      r'./',
      (r'https://github.com/GoogleCloudPlatform/explainable_ai_sdk/'
       r'blob/master/'))

# Read dependencies requirements.txt
with open('requirements.txt', 'r') as f:
  required_packages = f.read().splitlines()

setuptools.setup(
    name=__package_name__,
    description='Helper library for CAIP explanations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=required_packages,
    packages=setuptools.find_packages(),
    version=__version__,
    author='Google LLC',
    author_email='xai-dev@googlegroups.com',
)
