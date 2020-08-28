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

"""Tests for google3.third_party.explainable_ai_sdk.sdk.model.utils."""
import tensorflow as tf
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import constants
from explainable_ai_sdk.model import utils


class UtilsTest(tf.test.TestCase):

  def test_get_modality_input_list_map_no_inputs(self):
    md = explain_metadata.ExplainMetadata.from_dict(
        {'inputs': {}, 'outputs': {}, 'framework': 'tensorflow2'})
    modalities = utils.get_modality_input_list_map(md)
    self.assertEmpty(modalities)

  def test_get_modality_input_list_map_with_bof(self):
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
    modalities = utils.get_modality_input_list_map(md)
    self.assertLen(modalities[constants.ALL_MODALITY], 5)
    self.assertLen(modalities[explain_metadata.Modality.CATEGORICAL], 3)
    self.assertLen(modalities[constants.TABULAR_MODALITY], 4)
    self.assertLen(modalities[explain_metadata.Modality.IMAGE], 1)

if __name__ == '__main__':
  tf.test.main()
