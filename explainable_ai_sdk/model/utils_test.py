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

"""Tests for google3.third_party.explainable_ai_sdk.sdk.model.utils."""
import os
from absl.testing import parameterized
import mock

import numpy as np
import tensorflow as tf
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_modality_inputs_map_from_metadata_no_inputs(self):
    md = explain_metadata.ExplainMetadata.from_dict({
        'inputs': {},
        'outputs': {},
        'framework': 'xgboost'
    })
    modalities = utils.create_modality_inputs_map_from_metadata(md)
    self.assertEmpty(modalities)

  def test_create_modality_inputs_map_from_metadata_with_bof(self):
    md = explain_metadata.ExplainMetadata.from_dict({
        'inputs': {
            'class': {
                'input_tensor_name': 'a:0',
                'encoding': 'bag_of_features',
                'index_feature_mapping': ['x', 'y', 'z'],
                'modality': 'categorical'
            },
            'img': {
                'input_tensor_name': 't:0',
                'modality': 'image'
            },
            'numer': {
                'input_tensor_name': 'n:0',
                'modality': 'numeric'
            }
        },
        'outputs': [],
        'framework': 'tensorflow2'
    })
    modalities = utils.create_modality_inputs_map_from_metadata(md)
    self.assertLen(modalities[constants.ALL_MODALITY], 5)
    self.assertLen(modalities[explain_metadata.Modality.CATEGORICAL], 3)
    self.assertLen(modalities[constants.TABULAR_MODALITY], 4)
    self.assertLen(modalities[explain_metadata.Modality.IMAGE], 1)

  @mock.patch.object(explain_metadata.ExplainMetadata, 'from_file')
  @mock.patch.object(utils, '_get_deployment_uri', autospec=True)
  def test_fetch_explanation_metadata_valid_path(self, mock_deployment_uri,
                                                 mock_explain_md_from_file):
    mock_deployment_uri.return_value = 'gs://test_bucket/test_model'
    utils.fetch_explanation_metadata('model_endpoint_uri')
    mock_explain_md_from_file.assert_called_once_with(
        'gs://test_bucket/test_model/explanation_metadata.json')

  @mock.patch.object(utils, '_get_deployment_uri', autospec=True)
  def test_fetch_explanation_metadata_invalid_path(self, mock_deployment_uri):
    mock_deployment_uri.return_value = 'gcs://test_bucket/test_model'
    with self.assertRaisesRegex(
        ValueError, ('The deployment uri is not a valid GCS bucket')):
      utils.fetch_explanation_metadata('model_endpoint_uri')

  @parameterized.named_parameters(
      ('caip', 'uc-central1', False,
       'https://uc-central1-ml.googleapis.com/v1/m/1/e/4'),
      ('vertex', 'us-east1', True,
       'https://us-east1-prediction-aiplatform.googleapis.com/v1beta1/m/1/e/4'),
      ('caip_no_region', None, False,
       'https://ml.googleapis.com/v1/m/1/e/4'))
  def test_get_endpoint_uri(self, region, is_vertex, expected_uri):
    self.assertEqual(
        expected_uri,
        utils.get_endpoint_uri('m/1/e/4', region, is_vertex))

  @mock.patch.dict(os.environ,
                   {constants.AIP_ENDPOINT_OVERRIDE: 'https://overriden'})
  def test_get_endpoint_uri_env_variable(self):
    self.assertEqual(
        'https://overriden/v1/m/1/e/4',
        utils.get_endpoint_uri('m/1/e/4'))

  def test_encode_ndarray_as_b64str(self):
    array = np.array([[[0.1, 0.1, 0.1], [0.2, 0.2, 0.3]]])
    expected = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD5/ooooA//2Q=='
    self.assertEqual(
        expected,
        utils.encode_ndarray_as_b64str(array))

if __name__ == '__main__':
  tf.test.main()
