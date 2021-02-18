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


"""Tests for model_factory."""
import mock
import tensorflow.compat.v1 as tf
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import ai_platform_model
from explainable_ai_sdk.model import model_factory
from explainable_ai_sdk.model import utils


class ModelFactoryTest(tf.test.TestCase):

  def setUp(self):
    super(ModelFactoryTest, self).setUp()
    self.addCleanup(mock.patch.stopall)
    mock.patch.object(
        utils,
        'fetch_explanation_metadata',
        return_value=explain_metadata.ExplainMetadata(
            inputs=[], framework='xgboost')).start()

  def test_load_model_from_ai_platform(self):
    model_factory.register_caip_model(ai_platform_model.AIPlatformModel)
    model = model_factory.load_model_from_ai_platform(
        'fake_project', 'fake_model', 'fake_version')
    self.assertIsInstance(model, ai_platform_model.AIPlatformModel)
    self.assertEqual(
        'https://ml.googleapis.com/v1/projects/fake_project/'
        'models/fake_model/versions/fake_version',
        model._model_endpoint_uri)

  def test_load_model_from_ai_platform_without_version(self):
    model_factory.register_caip_model(ai_platform_model.AIPlatformModel)
    model = model_factory.load_model_from_ai_platform('fake_project',
                                                      'fake_model')
    self.assertIsInstance(model, ai_platform_model.AIPlatformModel)
    self.assertEqual(
        'https://ml.googleapis.com/v1/projects/fake_project/'
        'models/fake_model',
        model._model_endpoint_uri)

  def test_load_model_from_unified_ai_platform(self):
    model_factory.register_ucaip_model(ai_platform_model.AIPlatformModel)
    model = model_factory.load_model_from_unified_ai_platform(
        'fake_project', 'fake_region', 'fake_endpoint')
    self.assertIsInstance(model, ai_platform_model.AIPlatformModel)
    self.assertEqual(
        'https://fake_region-prediction-aiplatform.googleapis.com/v1beta1/'
        'projects/fake_project/locations/fake_region/endpoints/fake_endpoint',
        model._model_endpoint_uri)

  def test_load_model_from_ai_platform_not_implemented(self):
    # Make sure the registry is empty.
    model_factory._MODEL_REGISTRY = {}
    with self.assertRaises(NotImplementedError):
      model_factory.load_model_from_ai_platform('fake_project', 'fake_model',
                                                'fake_version')


if __name__ == '__main__':
  tf.test.main()
