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


"""Model classes for obtaining explanations."""
import json
import os
import re


from absl import logging
import google.auth.credentials

from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import configs
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import explanation
from explainable_ai_sdk.model import http_utils
from explainable_ai_sdk.model import model
from explainable_ai_sdk.model import utils


class AIPlatformModel(model.Model):
  """Class for models loaded from AI Platform."""

  def __init__(
      self,
      endpoint,
      credentials = None):
    """Constructing basic information of the model.

    Args:
      endpoint: an AI Platform model endpoint (i.e.,
        projects/<project_name>/models/<model_name>/versions/<version_name>)
      credentials: The OAuth2.0 credentials to use for GCP services.
    """
    self._credentials = credentials
    self._endpoint = endpoint
    self._explanation_metadata = self._get_explanation_metadata()
    self._modality_input_list_map = utils.get_modality_input_list_map(
        self._explanation_metadata)

  def _get_deployment_uri(self):
    """A method to get the depolyment uri of the model.

    Returns:
      A string uri of a gcs bucket.

    Raises:
      KeyError: This error will be raised if the 'deploymentUri' key is
        missing from the returned version information.
    """
    response = http_utils.make_get_request_to_ai_platform(
        self._endpoint, self._credentials)

    if 'deploymentUri' not in response:
      raise KeyError('There is no deploymentUri information in this version')

    return response['deploymentUri']

  def _get_explanation_metadata_uri(self):
    """A method to get the uri of explanation_metadata.json.

    The method will call the ml service first to get deployment uri,
    and then append explanation_metadata.json to the uri to return.

    Returns:
       A uri to explanation_metatdata.json.

    Raises:
      ValueError: will be raised if the deployment uri is not a valid
        gcs bucket uri.
    """
    gcs_uri = self._get_deployment_uri()
    match = re.search('gs://(?P<bucket_name>[^/]*)[/]*(?P<directory>.*)',
                      gcs_uri)
    if match:
      object_path = os.path.join(gcs_uri,
                                 'explanation_metadata.json')
      return object_path

    raise ValueError('The deployment uri is not a valid GCS bucket')

  def _get_explanation_metadata(self):
    """A method to get explanation metadata.

    The method will call the ml service first to get deployment uri,
    and then call the gcs to retrieve explanation metadata.json file.

    Returns:
       A dictionary of explanation metatdata.

    """
    explanation_md_uri = self._get_explanation_metadata_uri()

    md = explain_metadata.ExplainMetadata.from_file(explanation_md_uri)

    return md

  def predict(self,
              instances,
              timeout_ms = constants.DEFAULT_TIMEOUT
             ):
    """A method to call prediction services with given instances.

    Args:
       instances: A list of instances for getting predictions.
       timeout_ms: Timeout for each service call to the api (in milliseconds).

    Returns:
       A list of the dictionaries.
    """
    request_body = {'instances': instances}
    response = http_utils.make_post_request_to_ai_platform(
        self._endpoint + ':predict',
        request_body,
        self._credentials,
        timeout_ms)

    return response

  def explain(self,
              instances,
              params = None,
              timeout_ms = constants.DEFAULT_TIMEOUT
             ):
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
    """
    if params:
      logging.warn('Params can not be overriden in a remote model at the'
                   ' moment.')
    del params
    request_body = {'instances': instances}
    response = http_utils.make_post_request_to_ai_platform(
        self._endpoint + ':explain',
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
      exp_obj = explanation.Explanation.from_ai_platform_response(
          explanation_dict, instances[idx], self._modality_input_list_map)
      explanations.append(exp_obj)

    return explanations
