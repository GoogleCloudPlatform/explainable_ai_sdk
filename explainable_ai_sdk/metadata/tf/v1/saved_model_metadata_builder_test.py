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

"""Tests for saved_model_metadata_builder."""
import os
import tensorflow.compat.v1 as tf

from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.metadata.tf.v1 import saved_model_metadata_builder


class SavedModelMetadataBuilderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(SavedModelMetadataBuilderTest, cls).setUpClass()
    cls.sess = tf.Session(graph=tf.Graph())
    with cls.sess.graph.as_default():
      cls.x = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='inp')
      weights = tf.constant(1., shape=(10, 2), name='weights')
      bias_weight = tf.constant(1., shape=(2,), name='bias')
      cls.linear_layer = tf.add(tf.matmul(cls.x, weights), bias_weight)
      cls.prediction = tf.nn.relu(cls.linear_layer)
      # save the model
      cls.model_path = os.path.join(tf.test.get_temp_dir(), 'saved_model')
      builder = tf.saved_model.builder.SavedModelBuilder(cls.model_path)
      tensor_info_x = tf.saved_model.utils.build_tensor_info(cls.x)
      tensor_info_pred = tf.saved_model.utils.build_tensor_info(cls.prediction)
      tensor_info_lin = tf.saved_model.utils.build_tensor_info(cls.linear_layer)
      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'x': tensor_info_x},
              outputs={'y': tensor_info_pred},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))
      double_output_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'x': tensor_info_x},
              outputs={'y': tensor_info_pred, 'lin': tensor_info_lin},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      builder.add_meta_graph_and_variables(
          cls.sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants
              .DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
              'double': double_output_signature
          })
      builder.save()

  def test_get_metadata_correct_inputs(self):
    md_builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        self.model_path, tags=[tf.saved_model.tag_constants.SERVING])
    self.assertLen(md_builder.get_metadata()['inputs'], 1)
    self.assertLen(md_builder.get_metadata()['outputs'], 1)

  def test_get_metadata_double_output(self):
    md_builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        self.model_path, signature_name='double', outputs_to_explain=['lin'])
    self.assertLen(md_builder.get_metadata()['outputs'], 1)
    self.assertIn('lin', md_builder.get_metadata()['outputs'])

  def test_save_metadata(self):
    md_builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        self.model_path, tags=[tf.saved_model.tag_constants.SERVING])
    filepath = self.create_tempdir().full_path
    md_builder.save_metadata(filepath)
    self.assertTrue(
        os.path.exists(os.path.join(filepath, 'explanation_metadata.json')))

  def test_save_model_with_metadata_successfully(self):
    model_path = self.create_tempdir().full_path
    md_builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        self.model_path, tags=[tf.saved_model.tag_constants.SERVING])
    md_builder.save_model_with_metadata(model_path)
    md = explain_metadata.ExplainMetadata.from_file(
        os.path.join(model_path, 'explanation_metadata.json'))
    self.assertDictEqual(md.to_dict()['inputs'],
                         md_builder.get_metadata()['inputs'])
    self.assertDictEqual(md.to_dict()['outputs'],
                         md_builder.get_metadata()['outputs'])

  def test_constructor_incorrect_signature_name(self):
    with self.assertRaisesRegex(ValueError, 'Serving sigdef key .* not in '
                                            'the signature def.'):
      _ = saved_model_metadata_builder.SavedModelMetadataBuilder(
          self.model_path,
          tags=[tf.saved_model.tag_constants.SERVING],
          signature_name='incorrect_signature')

  def test_constructor_empty_tags(self):
    md_builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        self.model_path, tags=[])
    self.assertLen(md_builder.get_metadata()['inputs'], 1)
    self.assertLen(md_builder.get_metadata()['outputs'], 1)

  def test_constructor_multiple_outputs_to_explain(self):
    with self.assertRaisesRegex(ValueError, 'Only one output is supported'):
      _ = saved_model_metadata_builder.SavedModelMetadataBuilder(
          self.model_path, outputs_to_explain=['out1', 'out2'])

  def test_constructor_no_outputs_explain(self):
    with self.assertRaisesRegex(ValueError, 'The signature contains multiple '
                                            'outputs'):
      _ = saved_model_metadata_builder.SavedModelMetadataBuilder(
          self.model_path, signature_name='double')


if __name__ == '__main__':
  tf.test.main()
