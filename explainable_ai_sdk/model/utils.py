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

"""Utility functions for models.
"""
import collections
import os
import re
from typing import Dict, List, Optional

import google.auth.credentials
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import http_utils


def _get_deployment_uri(
    model_endpoint_uri: str,
    credentials: Optional[google.auth.credentials.Credentials] = None) -> str:
  """Gets the deployment uri of the model from AIP.

  Args:
    model_endpoint_uri: Full model endpoint uri.
    credentials: The OAuth2.0 credentials to use for GCP services.

  Returns:
    String uri of the model GCS bucket.

  Raises:
    KeyError: This error will be raised if the 'deploymentUri' key is
      missing from the returned version information.
  """
  response = http_utils.make_get_request_to_ai_platform(model_endpoint_uri,
                                                        credentials)

  if 'deploymentUri' not in response:
    raise KeyError('There is no deploymentUri information in this version')

  return response['deploymentUri']


def _extract_explanation_metadata_uri(gcs_uri: str) -> str:
  """Extracts the uri of explanation_metadata.json from model GCS folder.

  Args:
    gcs_uri: GCS uri of the saved model.

  Returns:
     URI to explanation_metadata.json.

  Raises:
    ValueError: Raised if the deployment uri is not a valid GCS bucket uri.
  """
  match = re.search('gs://(?P<bucket_name>[^/]*)[/]*(?P<directory>.*)', gcs_uri)
  if not match:
    raise ValueError('The deployment uri is not a valid GCS bucket')

  return os.path.join(gcs_uri, 'explanation_metadata.json')


def fetch_explanation_metadata(
    model_endpoint_uri: str,
    credentials: Optional[google.auth.credentials.Credentials] = None
) -> explain_metadata.ExplainMetadata:
  """Fetches explanation metadata from user's deployment uri.

  This method will call the AIP service first to get deployment uri,
  and then call the gcs to retrieve explanation metadata.json file.

  Args:
    model_endpoint_uri: Full model endpoint uri.
    credentials: The OAuth2.0 credentials to use for GCP services.

  Returns:
     Model's explanation metatdata.
  """
  explanation_md_uri = _extract_explanation_metadata_uri(
      _get_deployment_uri(model_endpoint_uri, credentials))

  return explain_metadata.ExplainMetadata.from_file(explanation_md_uri)


def create_modality_inputs_map_from_metadata(
    explain_md: explain_metadata.ExplainMetadata) -> Dict[str, List[str]]:
  """Gets a mapping between modality and input lists.

  Args:
    explain_md: ExplainMetadata object to collect modalities from.

  Returns:
     A dictionary that maps modality to a list of input names.
  """
  modality_input_list_map = collections.defaultdict(list)

  for input_metadata in explain_md.inputs:
    input_name = input_metadata.name
    input_modality = input_metadata.modality
    input_encoding = input_metadata.encoding
    if input_modality in explain_metadata.Modality.values():
      if (input_encoding == explain_metadata.Encoding.BAG_OF_FEATURES or
          input_encoding == explain_metadata.Encoding.BAG_OF_FEATURES_SPARSE):
        for feature_name in input_metadata.index_feature_mapping:
          modality_input_list_map[explain_metadata.Modality.CATEGORICAL].append(
              feature_name)
          modality_input_list_map[constants.ALL_MODALITY].append(feature_name)
          modality_input_list_map[constants.TABULAR_MODALITY].append(
              feature_name)
      else:
        if (input_modality in (explain_metadata.Modality.NUMERIC,
                               explain_metadata.Modality.CATEGORICAL)):
          modality_input_list_map[constants.TABULAR_MODALITY].append(input_name)
        modality_input_list_map[input_modality].append(input_name)
        modality_input_list_map[constants.ALL_MODALITY].append(input_name)
  return modality_input_list_map


def get_endpoint_uri(resource_path: str,
                     region: Optional[str] = None,
                     is_ucaip: bool = False) -> str:
  """Creates the endpoint URI string from given model parameters.

  Args:
    resource_path: Full model path on the model endpoint.
    region: Region of the model.
    is_ucaip: Boolean flag to indicate if the model is a uCAIP model.

  Returns:
    Full endpoint URI to issue a request to.
  """
  if os.getenv(constants.AIP_ENDPOINT_OVERRIDE):
    ai_platform_endpoint = os.getenv(constants.AIP_ENDPOINT_OVERRIDE)
    version = constants.CAIP_API_ENDPOINT_VERSION
  elif is_ucaip:
    if not region:
      raise ValueError('uCAIP models require model region.')
    ai_platform_endpoint = constants.UCAIP_PREDICTION_API_ENDPOINT.format(
        region=region)
    version = constants.UCAIP_API_ENDPOINT_VERSION
  else:
    if region:
      ai_platform_endpoint = constants.CAIP_API_REGION_ENDPOINT.format(
          region=region)
    else:
      ai_platform_endpoint = constants.CAIP_API_ENDPOINT
    version = constants.CAIP_API_ENDPOINT_VERSION

  return os.path.join(ai_platform_endpoint, version, resource_path)
