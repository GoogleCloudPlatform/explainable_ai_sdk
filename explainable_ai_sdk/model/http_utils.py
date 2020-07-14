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


"""HTTP-related util functions in SDK."""


import requests

import google.auth
import google.auth.credentials
import google.auth.transport.requests

from explainable_ai_sdk.model import constants


def _get_request_header(
    credentials = None
):
  """Gets a request header.

  Args:
    credentials: The credentials to use for GCP services.

  Returns:
    A header dict for requests.
  """
  # If credentials is not given, use the default credentials
  if credentials is None:
    credentials, _ = google.auth.default()

  # Refresh credentials in case it has been expired.
  auth_req = google.auth.transport.requests.Request()
  credentials.refresh(auth_req)

  headers = {}
  # Set user-agent for logging usages.
  headers['user-agent'] = constants.USER_AGENT_FOR_CAIP_TRACKING
  credentials.apply(headers)

  return headers


def make_get_request_to_ai_platform(
    uri_params_str,
    credentials = None,
    timeout_ms = constants.DEFAULT_TIMEOUT):
  """Makes a get request to AI Platform.

  Args:
    uri_params_str: A string representing uri parameters (e.g.,
      projects/<proj_name>/models/<model_name>/versions/<version_name>).
    credentials: The OAuth2.0 credentials to use for GCP services.
    timeout_ms: Timeout for each service call to the api (in milliseconds).

  Returns:
    Request results in json format.
  """
  headers = _get_request_header(credentials)
  uri = constants.CAIP_API_ENDPOINT + uri_params_str

  r = requests.get(uri, headers=headers, timeout=timeout_ms)
  return r.json()


def make_post_request_to_ai_platform(
    uri_params_str,
    request_body,
    credentials = None,
    timeout_ms = constants.DEFAULT_TIMEOUT):
  """Makes a post request to AI Platform.

  Args:
    uri_params_str: A string representing uri parameters (e.g.,
      projects/<proj_name>/models/<model_name>/versions/<version_name>).
    request_body: A dict for the request body
    credentials: The OAuth2.0 credentials to use for GCP services.
    timeout_ms: Timeout for each service call to the api (in milliseconds).

  Returns:
    Request results in json format.
  """
  headers = _get_request_header(credentials)
  uri = constants.CAIP_API_ENDPOINT + uri_params_str

  r = requests.post(uri, headers=headers, json=request_body, timeout=timeout_ms)
  return r.json()
