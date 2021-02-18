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


"""Model classes for obtaining explanations."""
import json

from typing import Any, Dict, List, Optional
from absl import logging
import google.auth.credentials

from explainable_ai_sdk.model import configs
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import explanation
from explainable_ai_sdk.model import http_utils
from explainable_ai_sdk.model import model

_CAIP_ATTRIBUTIONS_KEY = 'attributions_by_label'
_CAIP_FEATURE_ATTRS_KEY = 'attributions_by_label'
_UCAIP_ATTRIBUTIONS_KEY = 'attributions'
_UCAIP_FEATURE_ATTRS_KEY = 'featureAttributions'


class AIPlatformModel(model.Model):
  """Class for models loaded from AI Platform."""

  def __init__(
      self,
      model_endpoint_uri: str,
      credentials: Optional[google.auth.credentials.Credentials] = None,
      modality_to_inputs_map: Optional[Dict[str, str]] = None):
    """Constructs a model backed by an endpoint on AI Platform.

    Args:
      model_endpoint_uri: Full (Unified) AI Platform model path (i.e.,
        <region>-prediction-aiplatform.googleapis.com/projects/<project_name>/
        models/<model_name>/versions/<version_name>)
      credentials: The OAuth2.0 credentials to use for GCP services.
      modality_to_inputs_map: Dictionary from modalities to inputs specified in
        the metadata.
    """
    self._credentials = credentials
    self._model_endpoint_uri = model_endpoint_uri
    self._modality_to_inputs_map = modality_to_inputs_map

  def predict(self,
              instances: List[Any],
              timeout_ms: int = constants.DEFAULT_TIMEOUT
             ) -> List[Dict[Any, Any]]:
    """A method to call prediction services with given instances.

    Args:
       instances: A list of instances for getting predictions.
       timeout_ms: Timeout for each service call to the api (in milliseconds).

    Returns:
       A list of the dictionaries.
    """
    request_body = {'instances': instances}
    response = http_utils.make_post_request_to_ai_platform(
        self._model_endpoint_uri + ':predict',
        request_body,
        self._credentials,
        timeout_ms)

    return response

  def explain(self,
              instances: List[Any],
              params: configs.AttributionParameters = None,
              timeout_ms: int = constants.DEFAULT_TIMEOUT
             ) -> List[explanation.Explanation]:
    """A method to call explanation services with given instances.

    Args:
       instances: A list of instances for getting explanations.
       params: Overridable parameters for the explain call. Parameters can not
         be overriden in a remote model at the moment.
       timeout_ms: Timeout for each service call to the api (in milliseconds).

    Returns:
       A list of Explanation objects.

    Raises:
      ValueError: When explanation service fails, raise ValueError with the
        returned error message. This is likely due to details or formats of
        the instances are not correct.
      KeyError: When the explanation dictionary doesn't contain 'attributions'
        or 'attributions_by_label' in the response.
    """
    if params:
      logging.warn('Params can not be overriden in a remote model at the'
                   ' moment.')
    del params
    request_body = {'instances': instances}
    response = http_utils.make_post_request_to_ai_platform(
        self._model_endpoint_uri + ':explain',
        request_body,
        self._credentials,
        timeout_ms)

    if 'error' in response:
      error_msg = response['error']
      raise ValueError(('Explanation call failed. This is likely due to '
                        'incorrect instance formats. \nOriginal error '
                        'message: ' + json.dumps(error_msg)))

    explanations = []
    for idx, explanation_dict in enumerate(response['explanations']):
      if _CAIP_ATTRIBUTIONS_KEY in explanation_dict:  # CAIP response.
        attrs_key = _CAIP_ATTRIBUTIONS_KEY
        feature_attrs_key = _CAIP_FEATURE_ATTRS_KEY
        parse_fun = explanation.Explanation.from_ai_platform_response
      elif _UCAIP_ATTRIBUTIONS_KEY in explanation_dict:  # uCAIP response.
        attrs_key = _UCAIP_ATTRIBUTIONS_KEY
        feature_attrs_key = _UCAIP_FEATURE_ATTRS_KEY
        parse_fun = explanation.Explanation.from_unified_ai_platform_response
      else:
        raise KeyError(
            'Attribution keys are not present in the AI Platform response.')

      attrs = explanation_dict[attrs_key]
      if not self._modality_to_inputs_map:
        self._modality_to_inputs_map = _create_default_modality_map(
            next(iter(attrs))[feature_attrs_key])
      explanations.append(
          parse_fun(attrs, instances[idx], self._modality_to_inputs_map))

    return explanations


def _create_default_modality_map(
    attribution: Dict[str, Any]) -> Dict[str, List[str]]:
  """Creates default modality map for the given explanation dict."""
  return {constants.ALL_MODALITY: list(attribution.keys())}
