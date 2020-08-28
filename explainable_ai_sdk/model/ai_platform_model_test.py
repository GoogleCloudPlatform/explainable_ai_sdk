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


"""Tests for model."""
import mock
import numpy as np
import tensorflow.compat.v1 as tf
from explainable_ai_sdk.model import ai_platform_model
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import http_utils
from explainable_ai_sdk.model import utils


class AIPlatformModelTest(tf.test.TestCase):

  @mock.patch.object(ai_platform_model.AIPlatformModel,
                     '_get_explanation_metadata')
  @mock.patch.object(
      utils,
      'get_modality_input_list_map',
      return_value={})
  @mock.patch.object(
      http_utils, 'make_post_request_to_ai_platform', autospec=True)
  def test_predict(self, mock_post_request_func, mock_get_metadata,
                   mock_get_modality_map):
    mock_post_request_func.return_value = {'predictions': [0.5]}

    m = ai_platform_model.AIPlatformModel('fake_end_point')
    instances = [{'images_str': {'b64': u'fake_b64_str'}}]
    predictions = m.predict(instances)

    self.assertEqual(predictions['predictions'][0], 0.5)

  @mock.patch.object(
      utils,
      'get_modality_input_list_map',
      return_value={constants.ALL_MODALITY: ['data']})
  @mock.patch.object(ai_platform_model.AIPlatformModel,
                     '_get_explanation_metadata')
  @mock.patch.object(
      http_utils, 'make_post_request_to_ai_platform', autospec=True)
  def test_explain(self, mock_post_request_func, mock_get_metadata,
                   mock_get_modality_map):
    mock_post_request_func.return_value = {
        'explanations': [{
            'attributions_by_label': [{
                'attributions': {
                    'data': [0.01, 0.02]
                },
                'baseline_score': 0.0001,
                'example_score': 0.80,
                'label_index': 3,
                'output_name': 'probability'
            }]
        }]
    }

    m = ai_platform_model.AIPlatformModel('fake_end_point')
    instances = [{'input': [0.05]}]
    explanations = m.explain(instances)

    self.assertTrue(mock_post_request_func.called)

    tensor_dict = explanations[0].as_tensors()
    self.assertTrue(
        np.array_equal(tensor_dict['data'], np.asarray([0.01, 0.02])))

  @mock.patch.object(
      utils,
      'get_modality_input_list_map',
      return_value={constants.ALL_MODALITY: ['data']})
  @mock.patch.object(ai_platform_model.AIPlatformModel,
                     '_get_explanation_metadata')
  @mock.patch.object(
      http_utils, 'make_post_request_to_ai_platform', autospec=True)
  def test_explain_with_error(self, mock_post_request_func, mock_get_metadata,
                              mock_get_modality_map):
    mock_post_request_func.return_value = {'error': 'This is an error.'}

    m = ai_platform_model.AIPlatformModel('fake_end_point')
    instances = [{'input': [0.05]}]
    with self.assertRaisesRegex(
        ValueError,
        ('Explanation call failed. .*\n'
         'Original error message: "This is an error."')):
      m.explain(instances)


if __name__ == '__main__':
  tf.test.main()
