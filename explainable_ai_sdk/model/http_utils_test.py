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


"""Tests for http_utils."""

import mock
import requests
import tensorflow.compat.v1 as tf

from explainable_ai_sdk.model import http_utils


class MockResponse(object):

  def json(self):
    pass


class HttpUtilsTest(tf.test.TestCase):

  @mock.patch.object(requests, 'get', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_get_request_to_ai_platform(self, mock_request_header,
                                           mock_get_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=200)
    mock_response().json.return_value = 'results'
    mock_get_func.return_value = mock_response()

    res = http_utils.make_get_request_to_ai_platform('uri/test_uri')
    self.assertTrue(mock_get_func.called)
    self.assertEqual(res, 'results')

  @mock.patch.object(requests, 'get', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_get_request_to_ai_platform_404_error(
      self, mock_request_header, mock_get_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=404)
    mock_get_func.return_value = mock_response()

    with self.assertRaisesRegex(
        ValueError,
        ('Target URI .* returns HTTP 404 error.\n'
         'Please check if the project, model, and version names '
         'are given correctly.')):
      http_utils.make_get_request_to_ai_platform('uri/test_uri')

  @mock.patch.object(requests, 'get', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_get_request_to_ai_platform_other_errors(
      self, mock_request_header, mock_get_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=501)
    type(mock_response()).text = mock.PropertyMock(return_value='test error')
    mock_get_func.return_value = mock_response()

    with self.assertRaisesRegex(
        ValueError,
        ('Target URI .* returns HTTP 501 error.\n'
         'Please check the raw error message: \n'
         'test error')):
      http_utils.make_get_request_to_ai_platform('uri/test_uri')

  @mock.patch.object(requests, 'post', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_post_request_to_ai_platform(self, mock_request_header,
                                            mock_post_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=200)
    mock_response().json.return_value = 'results'
    mock_post_func.return_value = mock_response()

    res = http_utils.make_post_request_to_ai_platform('uri/test_uri',
                                                      {'data': 123})
    self.assertTrue(mock_post_func.called)
    self.assertEqual(res, 'results')

  @mock.patch.object(requests, 'post', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_post_request_to_ai_platform_404_error(
      self, mock_request_header, mock_post_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=404)
    mock_post_func.return_value = mock_response()

    with self.assertRaisesRegex(
        ValueError,
        ('Target URI .* returns HTTP 404 error.\n'
         'Please check if the project, model, and version names '
         'are given correctly.')):
      http_utils.make_post_request_to_ai_platform('uri/test_uri',
                                                  {'data': 123})

  @mock.patch.object(requests, 'post', autospec=True)
  @mock.patch.object(http_utils, '_get_request_header', autospec=True)
  def test_make_post_request_to_ai_platform_other_erors(
      self, mock_request_header, mock_post_func):
    mock_request_header.return_value = {}

    mock_response = mock.Mock(spec=MockResponse)
    type(mock_response()).status_code = mock.PropertyMock(return_value=403)
    type(mock_response()).text = mock.PropertyMock(return_value='test error')
    mock_post_func.return_value = mock_response()

    with self.assertRaisesRegex(
        ValueError,
        ('Target URI .* returns HTTP 403 error.\n'
         'Please check the raw error message: \n'
         'test error')):
      http_utils.make_post_request_to_ai_platform('uri/test_uri',
                                                  {'data': 123})


if __name__ == '__main__':
  tf.test.main()
