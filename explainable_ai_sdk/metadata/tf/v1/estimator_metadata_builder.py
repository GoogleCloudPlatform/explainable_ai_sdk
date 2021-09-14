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


"""Metadata builder helper for models built with Estimator API.

This module utilizes monkey_patch_utils to monkey patch certain functions in
feature columns and estimators to observe created tensors. Then, explanation
metadata is created from these groups of tensors.

Users of EstimatorMetadataBuilder need to instantiate the builder with an
estimator instance, base feature columns they want explanations for, serving
input function they want to export the model with, output key, and other kwargs
to be passed to export_saved_model function of estimators.

Example usage is as follows:
  classifier.train(...)
  md_builder = EstimatorMetadataBuilder(classifier, columns, serving_input_fn)
  md_builder.save_model_with_metadata(export_path)

Caveats:
  - If an input is encoded multiple times, we don't include any encoded tensors
    in the metadata. If those inputs are sparse, users need to use Sampled
    Shapley. When we have the support for attributing from multiple encoded
    tensors to a single input for methods like IG, then we can update this
    module to include encoded tensors in input metadata.

  - Having multiple input tensors for a feature column is a rare case. But when
    it happens, we create two input metadata for that feature because there are
    multiple sets of input tensors and encoded tensors in parallel.

  - There is no get_metadata() function because these tensors can be observed
    only when the estimator is being saved.
"""
import enum
import shutil
import tempfile
from typing import Dict, Text, List, Any, Set, Callable, Optional, Union

import tensorflow.compat.v1 as tf
from tensorflow.python.feature_column import feature_column_v2 as fc2  
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.metadata import constants
from explainable_ai_sdk.metadata import metadata_builder
from explainable_ai_sdk.metadata import utils
from explainable_ai_sdk.metadata.tf.v1 import monkey_patch_utils


@enum.unique
class DuplicateAction(enum.Enum):
  """Enum for action to take in case of duplicate inputs."""
  NOTHING = enum.auto()
  DROP = enum.auto()
  GROUP = enum.auto()


class EstimatorMetadataBuilder(metadata_builder.MetadataBuilder):
  """Class for generating metadata for models built with Estimator API."""

  def __init__(self,
               estimator: tf.estimator.Estimator,
               feature_columns: List[fc2.FeatureColumn],
               serving_input_fn: Callable[..., Any],
               output_key: Optional[Text] = None,
               baselines: Optional[Dict[Text, List[Any]]] = None,
               input_mds: Optional[List[explain_metadata.InputMetadata]] = None,
               duplicate_feature_treatment: Union[
                   str, DuplicateAction] = DuplicateAction.NOTHING,
               **kwargs):
    """Initialize an EstimatorMetadataBuilder.

    Args:
      estimator: Estimator instance to observe and save.
      feature_columns: A group of feature columns to export metadata for. These
        feature columns need to be basic feature columns and not derived
        columns such as embedding, indicator, bucketized.
      serving_input_fn: Serving input function to be used when exporting the
        model.
      output_key: Output key to find the model's relevant output tensors. Some
        valid values are logits, probabilities. If not provided, will default to
        logits and regression outputs.
      baselines: Baseline values for each input feature. The key name is the
        feature name, and the value represents the baselines. The value is
        specified as a list because multiple baselines can be supplied for the
        features.
      input_mds: Additional inputs to the Metadata. If this provided, it will
        be added in addition to inputs from feature_columns.
      duplicate_feature_treatment: Treatment for duplicate inputs for the same
        feature column. Either provide a DuplicateAction enum or a string where
        the possible values are {'nothing', 'drop', 'group'}. If set
        to 'nothing', all are included with unique suffix added to input names
        to disambiguate. If set to 'drop', only one of the inputs will be
        retained. If set to 'group', all inputs will be grouped.
      **kwargs: Any keyword arguments to be passed to export_saved_model.
        add_meta_graph() function.
    """
    if not isinstance(estimator, tf.estimator.Estimator):
      raise ValueError('A valid estimator needs to be provided.')
    self._estimator = estimator
    if not feature_columns:
      raise ValueError('feature_columns cannot be empty.')
    if isinstance(duplicate_feature_treatment, str):
      if duplicate_feature_treatment.upper() not in [a.name
                                                     for a in DuplicateAction]:
        raise ValueError('Unrecognized treatment option for duplicates:'
                         f'{duplicate_feature_treatment}.')
      duplicate_feature_treatment = DuplicateAction[
          duplicate_feature_treatment.upper()]
    self._feature_columns = feature_columns
    self._input_mds = input_mds
    self._output_key = output_key
    self._baselines = baselines
    self._serving_input_fn = serving_input_fn
    self._duplicate_action = duplicate_feature_treatment
    self._save_args = kwargs
    self._metadata = None

  def _get_input_tensor_names_for_metadata(
      self,
      feature_tensors: monkey_patch_utils.FeatureTensors) -> Dict[Text, Text]:
    """Returns a dictionary of tensor names for given FeatureTensors object."""
    input_md = {}
    if isinstance(feature_tensors.input_tensor, tf.Tensor):
      input_md['input_tensor_name'] = feature_tensors.input_tensor.name
    else:  # IdWeightPair -- sparse tensor
      sparse_ids, weights = feature_tensors.input_tensor

      input_md['input_tensor_name'] = sparse_ids.values.name
      input_md['indices_tensor_name'] = sparse_ids.indices.name
      input_md['dense_shape_tensor_name'] = sparse_ids.dense_shape.name
      if weights:
        input_md['weight_values_name'] = weights.values.name
        input_md['weight_indices_name'] = weights.indices.name
        input_md['weight_dense_shape_name'] = weights.dense_shape.name
    return input_md

  def _get_encoded_tensor_names_for_metadata(
      self,
      feature_tensors: monkey_patch_utils.FeatureTensors) -> Dict[Text, Text]:
    """Returns encoded tensor names only if there is a single encoded tensor."""
    input_md = {}
    if len(feature_tensors.encoded_tensors) == 1:
      # Currently, encoding is always the combined embedding.
      input_md['encoded_tensor_name'] = feature_tensors.encoded_tensors[0].name
      input_md['encoding'] = 'combined_embedding'
    return input_md

  def _create_input_metadata(
      self,
      features_dict: Dict[Text, List[monkey_patch_utils.FeatureTensors]],
      crossed_columns: Set[Text],
      desired_columns: List[Text]):
    """Creates and returns a list of InputMetadata.

    Args:
      features_dict: Dictionary from feature name to FeatureTensors class.
      crossed_columns: A set of crossed column names.
      desired_columns: A list of feature column names. Only the columns in
        this list will be added to input metadata.

    Returns:
      A list of InputMetadata.
    """
    input_mds = []
    if self._input_mds is not None:
      input_mds = self._input_mds
    input_names_processed = set([i.name for i in input_mds])
    for fc_name, tensor_groups in features_dict.items():
      if fc_name in desired_columns and fc_name not in input_names_processed:
        for tensor_group in tensor_groups:
          input_md = self._get_input_tensor_names_for_metadata(tensor_group)
          if fc_name not in crossed_columns:
            input_md.update(
                self._get_encoded_tensor_names_for_metadata(tensor_group))
          input_md['name'] = fc_name
          if self._baselines:
            input_md['input_baselines'] = self._baselines.get(fc_name, None)
          if (len(tensor_groups) == 1 or
              self._duplicate_action == DuplicateAction.DROP):
            input_mds.append(explain_metadata.InputMetadata(**input_md))
            break  # Skip other tensor_groups.
          # There are multiple inputs for the same feature column.
          # Append part of the tensor name until the first '/'. This usually
          # specifies what kind of model it is: linear or dnn.
          input_tensor_name = str(input_md['input_tensor_name'])
          suffix = input_tensor_name.split('/')[0]
          input_md['name'] = '%s_%s' % (fc_name, suffix)
          if self._duplicate_action == DuplicateAction.GROUP:
            input_md['group_name'] = fc_name
          input_mds.append(explain_metadata.InputMetadata(**input_md))
    return input_mds

  def _create_output_metadata(self, output_dict: Dict[Text, tf.Tensor]):
    """Creates and returns a list of OutputMetadata.

    Args:
      output_dict: Dictionary from tf.feature_columns to list of dense tensors.

    Returns:
      A list of OutputMetadata.
    """
    return [explain_metadata.OutputMetadata(name, tensor.name)
            for name, tensor in output_dict.items()]

  def _create_metadata_from_tensors(
      self,
      features_dict: Dict[Text, List[monkey_patch_utils.FeatureTensors]],
      crossed_columns: Set[Text],
      desired_columns: List[Text],
      output_dict: Dict[Text, tf.Tensor]) -> explain_metadata.ExplainMetadata:
    """Creates metadata from given tensor information.

    Args:
      features_dict: Dictionary from feature name to FeatureTensors class.
      crossed_columns: A set of crossed column names.
      desired_columns: A list of feature column names. Only the columns in
        this list will be added to input metadata.
      output_dict: Dictionary from tf.feature_columns to list of dense tensors.

    Returns:
      A dictionary that abides to explanation metadata.
    """
    return explain_metadata.ExplainMetadata(
        inputs=self._create_input_metadata(
            features_dict,
            crossed_columns,
            desired_columns),
        outputs=self._create_output_metadata(output_dict),
        framework='Tensorflow',
        tags=[constants.METADATA_TAG])

  def save_model_with_metadata(self, file_path: str) -> str:
    """Saves the model and the generated metadata to the given file path.

    New metadata will not be generated for each call to this function since an
    Estimator is static. Calling this function with different paths will save
    the model and the same metadata to all paths.

    Args:
      file_path: Path to save the model and the metadata. It can be a GCS bucket
        or a local folder. The folder needs to be empty.

    Returns:
      Full file path where the model and the metadata are written.
    """
    monkey_patcher = monkey_patch_utils.EstimatorMonkeyPatchHelper()
    with monkey_patcher.exporting_context(self._output_key):
      model_path = self._estimator.export_saved_model(
          file_path, self._serving_input_fn, **self._save_args)

    if not self._metadata:
      self._metadata = self._create_metadata_from_tensors(
          monkey_patcher.feature_tensors_dict,
          monkey_patcher.crossed_columns,
          [fc.name for fc in self._feature_columns],
          monkey_patcher.output_tensors_dict)
    utils.write_metadata_to_file(self._metadata.to_dict(), model_path)
    return model_path.decode('utf-8')

  def get_metadata(self) -> Dict[str, Any]:
    """Returns the current metadata as a dictionary.

    Since metadata creation is somewhat costly. The one already created is
    returned as a dictionary. If it hasn't been created, the model is saved to a
    temporary folder as the metadata tensors are created during model save. That
    temporary folder is deleted afterwards.
    """
    if self._metadata:
      return self._metadata.to_dict()
    temp_location = tempfile.gettempdir()
    model_path = self.save_model_with_metadata(temp_location)
    shutil.rmtree(model_path)
    return self._metadata.to_dict()
