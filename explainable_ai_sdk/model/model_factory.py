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


"""Factory for the SDK model class.

Currently, there are two kinds of models: local and remote. Model classes can be
registered to serve as remote or local.
"""
import os
from typing import Optional, Type, Dict
import google.auth.credentials
from explainable_ai_sdk.model import configs
from explainable_ai_sdk.model import model as model_lib
from explainable_ai_sdk.model import utils

_CAIP_MODEL_KEY = 'caip_model'
_UCAIP_MODEL_KEY = 'ucaip_model'
_LOCAL_MODEL_KEY = 'local_model'
_MODEL_REGISTRY = {}


def load_model_from_ai_platform(
    project: str,
    model: str,
    version: Optional[str] = None,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    region: Optional[str] = None,
    input_modalities: Optional[Dict[str, str]] = None
) -> model_lib.Model:
  """Loads a model from Cloud AI Platform.

  Args:
    project: AI Platform project name.
    model: AI Platform Prediction model name.
    version: Version of the given model. If not given, it will load the
      default version for the model.
    credentials: The OAuth2.0 credentials to use for GCP services.
    region: GCP Region for the deployed model.
    input_modalities: Dictionary mapping from modalities to input names in the
      explain metadata. For example {'numeric': ['input1', 'input2'], 'all':
      ['input1', 'input2']}. All inputs must be collected under the 'all' key.
      If None, modalities will be inferred from the model metadata.

  Returns:
     A model object

  Raises:
    NotImplementedError: If there are no registered remote models.
  """
  if _CAIP_MODEL_KEY not in _MODEL_REGISTRY:
    available_models = ', '.join(_MODEL_REGISTRY)
    raise NotImplementedError('There are no implementations for CAIP models. '
                              f'Avilable models are: {{{available_models}}}.')
  resouce_path = os.path.join('projects', project, 'models', model)
  if version:
    resouce_path = os.path.join(resouce_path, 'versions', version)

  model_endpoint_uri = utils.get_endpoint_uri(resouce_path, region, False)
  if not input_modalities:
    input_modalities = utils.create_modality_inputs_map_from_metadata(
        utils.fetch_explanation_metadata(model_endpoint_uri, credentials))

  return _MODEL_REGISTRY[_CAIP_MODEL_KEY](model_endpoint_uri, credentials,
                                          input_modalities)


def load_model_from_unified_ai_platform(
    project: str,
    region: str,
    endpoint_id: str,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    input_modalities: Optional[Dict[str, str]] = None
) -> model_lib.Model:
  """Loads a model from Unified Cloud AI Platform.

  Args:
    project: AI Platform project name.
    region: GCP Region for the deployed model.
    endpoint_id: Version of the given model. If not given, it will load the
      default version for the model.
    credentials: The OAuth2.0 credentials to use for GCP services.
    input_modalities: Dictionary mapping from modalities to input names in the
      explain metadata. For example {'numeric': ['input1', 'input2'], 'all':
      ['input1', 'input2']}. All inputs must be collected under the 'all' key.
      If None, a default metadata of {'all': [...]} will be created.

  Returns:
     A model object

  Raises:
    NotImplementedError: If there are no registered remote models.
  """
  if _UCAIP_MODEL_KEY not in _MODEL_REGISTRY:
    available_models = ', '.join(_MODEL_REGISTRY)
    raise NotImplementedError('There are no implementations for uCAIP models. '
                              f'Avilable models are: {{{available_models}}}.')
  resource_path = os.path.join('projects', project, 'locations', region,
                               'endpoints', endpoint_id)
  return _MODEL_REGISTRY[_UCAIP_MODEL_KEY](utils.get_endpoint_uri(
      resource_path, region, True), credentials, input_modalities)


def load_model_from_local_path(
    model_path: str, config: configs.AttributionConfig) -> model_lib.Model:
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


def register_caip_model(registered_class: Type[model_lib.Model]) -> None:
  """Register given AI Platform model class."""
  _MODEL_REGISTRY[_CAIP_MODEL_KEY] = registered_class


def register_ucaip_model(registered_class: Type[model_lib.Model]) -> None:
  """Register given Unified AI Platform model class."""
  _MODEL_REGISTRY[_UCAIP_MODEL_KEY] = registered_class


def register_local_model(registered_class: Type[model_lib.Model]) -> None:
  """Register given local class."""
  _MODEL_REGISTRY[_LOCAL_MODEL_KEY] = registered_class
