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


"""Tests for model_factory."""
import mock
import tensorflow.compat.v1 as tf
from explainable_ai_sdk.model import ai_platform_model
from explainable_ai_sdk.model import model_factory


class ModelFactoryTest(tf.test.TestCase):

  @mock.patch.object(
      ai_platform_model.AIPlatformModel, '__init__', autospec=True)
  def test_load_model_from_ai_platform(self, mock_constructor):
    model_factory.register_remote_model(ai_platform_model.AIPlatformModel)
    mock_constructor.return_value = None
    model_factory.load_model_from_ai_platform('fake_project', 'fake_model',
                                              'fake_version')
    self.assertTrue(mock_constructor.called)

  @mock.patch.object(
      ai_platform_model.AIPlatformModel, '__init__', autospec=True)
  def test_load_model_from_ai_platform_without_version(self, mock_constructor):
    model_factory.register_remote_model(ai_platform_model.AIPlatformModel)

    def side_effect(self, endpoint, unused_credentials):
      self.endpoint = endpoint

    mock_constructor.side_effect = side_effect
    model = model_factory.load_model_from_ai_platform('fake_project',
                                                      'fake_model')
    expected_endpoint = 'projects/fake_project/models/fake_model'
    self.assertTrue(model.endpoint, expected_endpoint)

  def test_load_model_from_ai_platform_not_implemented(self):
    # Make sure the registry is empty.
    model_factory._MODEL_REGISTRY = {}
    with self.assertRaises(NotImplementedError):
      model_factory.load_model_from_ai_platform('fake_project', 'fake_model',
                                                'fake_version')


if __name__ == '__main__':
  tf.test.main()
