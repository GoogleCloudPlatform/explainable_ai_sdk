# Copyright 2020 Google LLC
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


"""Factory for the SDK model class.

Currently, there are two kinds of models: local and remote. Model classes can be
registered to serve as remote or local.
"""
import os

import google.auth.credentials
from explainable_ai_sdk.model import configs
from explainable_ai_sdk.model import model as model_lib

_REMOTE_MODEL_KEY = 'remote'
_LOCAL_MODEL_KEY = 'local'
_MODEL_REGISTRY = {}


def load_model_from_ai_platform(
    project,
    model,
    version = None,
    credentials = None
):
  """Loads a model from Cloud AI Platform.

  Args:
    project: an AI Platform project name.
    model: an AI Platform Prediction model name.
    version: a version of the given model. If not given, it will load the
      default version for the model.
    credentials: The OAuth2.0 credentials to use for GCP services.

  Returns:
     A model object

  Raises:
    NotImplementedError: If there are no registered remote models.
  """
  if _REMOTE_MODEL_KEY not in _MODEL_REGISTRY:
    raise NotImplementedError('There are no implementations of remote model.')
  endpoint = os.path.join('projects', project, 'models', model)
  if version:
    endpoint = os.path.join(endpoint, 'versions', version)
  return _MODEL_REGISTRY[_REMOTE_MODEL_KEY](endpoint, credentials)


def load_model_from_local_path(
    model_path, config):
  """Loads a model based on a local model's path and attribution config.

  Args:
    model_path: A path that contains a saved model.
    config: Configuration parameters for attribution method.

  Returns:
     A model object.

  Raises:
    NotImplementedError: If there are no registered local models.
  """
  if _LOCAL_MODEL_KEY not in _MODEL_REGISTRY:
    raise NotImplementedError('There are no implementations of local model.')
  return _MODEL_REGISTRY[_LOCAL_MODEL_KEY](model_path, config)


def register_remote_model(registered_class):
  """Register given remote class."""
  _MODEL_REGISTRY[_REMOTE_MODEL_KEY] = registered_class


def register_local_model(registered_class):
  """Register given local class."""
  _MODEL_REGISTRY[_LOCAL_MODEL_KEY] = registered_class
