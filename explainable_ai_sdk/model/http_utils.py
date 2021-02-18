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


"""HTTP-related util functions in SDK."""

from typing import Any, Dict, Optional

import google.auth
import google.auth.credentials
import google.auth.transport.requests
import requests

from explainable_ai_sdk.model import constants


def _get_request_header(
    credentials: Optional[google.auth.credentials.Credentials] = None
) -> Dict[str, Any]:
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


def _handle_ai_platform_response(uri, response):
  """Handle response to AI platform from both get/post calls.

  Args:
    uri: Request uri.
    response: Response from the request.
  Returns:
    Request results in json format.
  Raises:
    ValueError: When the request fails, the ValueError will be raised with
      either 404 error or the raw errors.
  """
  if response.status_code == 200:
    return response.json()
  elif response.status_code == 404:
    raise ValueError(('Target URI {} returns HTTP 404 error.\n'
                      'Please check if the project, model, and version names '
                      'are given correctly.'
                      ).format(uri))
  raise ValueError(('Target URI {} returns HTTP {} error.\n'
                    'Please check the raw error message: \n{}'
                    ).format(uri, response.status_code, response.text))


def make_get_request_to_ai_platform(
    uri: str,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    timeout_ms: int = constants.DEFAULT_TIMEOUT):
  """Makes a get request to AI Platform.

  Args:
    uri: URI of the model resource.
    credentials: The OAuth2.0 credentials to use for GCP services.
    timeout_ms: Timeout for each service call to the api (in milliseconds).

  Returns:
    Request results in json format.
  """
  headers = _get_request_header(credentials)

  r = requests.get(uri, headers=headers, timeout=timeout_ms)
  return _handle_ai_platform_response(uri, r)


def make_post_request_to_ai_platform(
    uri: str,
    request_body: Dict[str, Any],
    credentials: Optional[google.auth.credentials.Credentials] = None,
    timeout_ms: int = constants.DEFAULT_TIMEOUT):
  """Makes a post request to AI Platform.

  Args:
    uri: URI of the model resource.
    request_body: A dict for the request body
    credentials: The OAuth2.0 credentials to use for GCP services.
    timeout_ms: Timeout for each service call to the api (in milliseconds).

  Returns:
    Request results in json format.
  """
  headers = _get_request_header(credentials)

  r = requests.post(uri, headers=headers, json=request_body, timeout=timeout_ms)
  return _handle_ai_platform_response(uri, r)
