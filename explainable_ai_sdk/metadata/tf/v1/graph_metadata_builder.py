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


"""GraphMetadataBuilder helps write metadata for models built with low-level TF API.

To use this builder module, one should add tensors and other parameters to build
explanation metadata for Explainable AI service.

Currently, XAI supports only single output. So the users are expected to add
any number of inputs (tabular, image, text) they wish, but can only add one
output tensor.

At any time, current metadata can be fetched via get_metadata() function. Once,
adding inputs and output is complete, metadata can be exported as a file along
with a saved model via save_model_with_metadata(...) function. This folder is
ready to be deployed to AI Platform with explainability flags.
"""
from typing import Dict, Text, Optional, List, Any, Set, Union
import tensorflow.compat.v1 as tf
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.metadata import constants
from explainable_ai_sdk.metadata import metadata_builder
from explainable_ai_sdk.metadata import parameters
from explainable_ai_sdk.metadata import utils as common_utils
from explainable_ai_sdk.metadata.tf.v1 import utils


class GraphMetadataBuilder(metadata_builder.MetadataBuilder):
  """Class for generating metadata for models built with low-level TF API."""

  def __init__(self,
               session: tf.Session = None,
               serving_inputs: Optional[Dict[Text, tf.Tensor]] = None,
               serving_outputs: Optional[Dict[Text, tf.Tensor]] = None,
               tags: Set[Text] = (tf.saved_model.tag_constants.SERVING,),
               **kwargs):
    """Initializes a GraphMetadataBuilder object.

    Args:
      session: tf.Session the model is being built. If not provided, a new
        session with the default graph will be created.
      serving_inputs: A dictionary mapping from serving key to corresponding
        input tensors. If not provided or empty, model input tensors will be
        used.
      serving_outputs: A dictionary mapping from serving key to model outputs.
        If not provided or empty, model output will be used.
      tags: The set of tags to annotate the meta graph def with.
      **kwargs: Any keyword arguments to be passed to saved model builder's
        add_meta_graph() function.
    """
    self._inputs, self._outputs = {}, {}
    self._session = session if session else tf.Session()
    self._serving_inputs = serving_inputs
    self._serving_outputs = serving_outputs
    self._tags = tags
    self._saved_model_args = kwargs

  def _add_input_metadata(
      self,
      input_tensor: tf.Tensor,
      name: Optional[Text] = None,
      encoded_tensor: Optional[tf.Tensor] = None,
      encoding: Optional[Text] = explain_metadata.Encoding.IDENTITY,
      input_baselines: Optional[List[Any]] = None,
      encoded_baselines: Optional[List[Any]] = None,
      modality: Optional[Text] = None,
      visualization: Optional[Union[Dict[str, str],
                                    parameters.VisualizationParameters]] = None,
      index_feature_mapping: Optional[List[Any]] = None,
      domain: Optional[parameters.DomainInfo] = None):
    """Creates an InputMetadata object.

    Args:
      input_tensor: Input tensor for the metadata.
      name: Metadata name for the given input.
      encoded_tensor: Encoded tensor if a tensor representing categorical input
        is encoded to another tensor.
      encoding: Encoding type. One of the values in explain_metadata.Encoding.
      input_baselines: A list of baselines for the input tensor.
      encoded_baselines: A list of baselines for the encoded tensor.
      modality: Modality of the input. One of the values in
        explain_metadata.Modality.
      visualization: Visualization parameters for image inputs. It can either be
        a dictionary of inputs or VisualizationParameters.
      index_feature_mapping: A list of feature names for each index in the input
        tensor.
      domain: DomainInfo object specifying the range of the input feature.
    """
    input_name = name if name else input_tensor.op.name
    encoded_tensor_name = (encoded_tensor.name if encoded_tensor is not None
                           else None)
    if input_tensor.name in self._inputs:
      raise ValueError('Input tensor %s already exists' % input_name)
    if input_name in [input_md.name for input_md in self._inputs.values()]:
      raise ValueError('Input name %s already exists' % input_name)
    domain_dict = domain.asdict() if domain else None
    if (visualization and
        isinstance(visualization, parameters.VisualizationParameters)):
      visualization = visualization.asdict()
    self._inputs[input_tensor.name] = explain_metadata.InputMetadata(
        name=input_name,
        input_tensor_name=input_tensor.name,
        encoded_tensor_name=encoded_tensor_name,
        encoding=encoding,
        input_baselines=input_baselines,
        encoded_baselines=encoded_baselines,
        modality=modality,
        visualization=visualization,
        index_feature_mapping=index_feature_mapping,
        domain=domain_dict)

  def add_numeric_metadata(self,
                           input_tensor: tf.Tensor,
                           name: Optional[Text] = None,
                           input_baselines: Optional[List[Any]] = None,
                           index_feature_mapping: Optional[List[Any]] = None):
    """Adds a numeric (float) tensor as input metadata.

    Args:
      input_tensor: A float tensor representing the input.
      name: Unique friendly name for this tensor. Returned attributions will be
        keyed with this name.
      input_baselines: A list of baseline values. Each baseline value can be a
        single entity or of the same shape as the input_tensor (except for the
        batch dimension).
      index_feature_mapping: A list of feature names for each index in the input
        tensor.
    """
    if index_feature_mapping:
      encoding = explain_metadata.Encoding.BAG_OF_FEATURES
    else:
      encoding = explain_metadata.Encoding.IDENTITY
    self._add_input_metadata(
        input_tensor,
        name,
        input_baselines=input_baselines,
        encoding=encoding,
        index_feature_mapping=index_feature_mapping,
        modality=explain_metadata.Modality.NUMERIC)

  def add_categorical_metadata(self,
                               input_tensor: tf.Tensor,
                               encoded_tensor: tf.Tensor,
                               encoding: Text,
                               name: Optional[Text] = None,
                               input_baselines: Optional[List[Any]] = None,
                               encoded_baselines: Optional[List[Any]] = None):
    """Adds a categorical input as input metadata.

    Args:
      input_tensor: Tensor to be treated as model feature.
      encoded_tensor: encoded_tensor if the given input_tensor is encoded.
      encoding: Encoding type if encoded_tensor is provided. Possible values are
        {identity, bag_of_features, bag_of_features_sparse, indicator,
        combined_embedding, concat_embedding}.
      name: Unique friendly name for this tensor. Returned attributions will be
        keyed with this name.
      input_baselines: A list of baseline values. Each baseline value can be a
        single entity or of the same shape as the input_tensor (except for the
        batch dimension).
      encoded_baselines: A list of baseline values for encoded tensor. Each
        baseline value can be a single entity or of the same shape as the
        input_tensor (except for the batch dimension).
    """
    self._add_input_metadata(
        input_tensor,
        name,
        encoded_tensor,
        encoding,
        input_baselines,
        encoded_baselines,
        modality=explain_metadata.Modality.CATEGORICAL)

  def add_image_metadata(
      self,
      input_tensor: tf.Tensor,
      name: Optional[str] = None,
      input_baselines: Optional[List[Any]] = None,
      visualization: Optional[Union[Dict[str, str],
                                    parameters.VisualizationParameters]] = None,
      domain: Optional[parameters.DomainInfo] = None):
    """Adds a new tensor representing image as input metadata.

    Args:
      input_tensor: Tensor to be treated as model feature.
      name: Unique friendly name for this tensor. Returned attributions will be
        keyed with this name.
      input_baselines: A list of baseline values. Each baseline value can be a
        single entity or of the same shape as the input_tensor (except for the
        batch dimension).
      visualization: Either a dictionary of visualization parameters or
        VisualizationParameters instance. Using VisualizationParameters is
        recommended. If None, a default visualization will be selected based on
        the explanation method (IG/XRAI).
      domain: DomainInfo object specifying the range of the input feature.
    """
    self._add_input_metadata(
        input_tensor,
        name,
        input_baselines=input_baselines,
        modality=explain_metadata.Modality.IMAGE,
        visualization=visualization,
        domain=domain)

  def add_text_metadata(
      self,
      input_tensor: tf.Tensor,
      encoded_tensor: Optional[tf.Tensor] = None,
      encoding: Optional[Text] = explain_metadata.Encoding.IDENTITY,
      name: Optional[Text] = None,
      input_baselines: Optional[List[Any]] = None,
      encoded_baselines: Optional[List[Any]] = None):
    """Adds a new tensor representing text input as input metadata.

    Args:
      input_tensor: Tensor to be treated as model feature.
      encoded_tensor: encoded_tensor if the given input_tensor is encoded.
      encoding: Encoding type if encoded_tensor is provided. Possible values are
        {identity, bag_of_features, bag_of_features_sparse, indicator,
        combined_embedding, concat_embedding}.
      name: Unique friendly name for this tensor. Returned attributions will be
        keyed with this name.
      input_baselines: A list of baseline values. Each baseline value can be a
        single entity or of the same shape as the input_tensor (except for the
        batch dimension).
      encoded_baselines: A list of baseline values for encoded tensor. Each
        baseline value can be a single entity or of the same shape as the
        input_tensor (except for the batch dimension).
    """
    self._add_input_metadata(
        input_tensor, name, encoded_tensor, encoding, input_baselines,
        encoded_baselines, modality=explain_metadata.Modality.TEXT)

  def add_output_metadata(self,
                          output_tensor: tf.Tensor,
                          name: Optional[Text] = None):
    """Adds output tensor as output metadata.

    Only one output metadata can be added.

    Args:
      output_tensor: Output tensors to get the explanations for. Needs to be a
        tensor of float type, such as probabilities, logits.
      name: Unique friendly name for the output.
    """
    if self._outputs:
      raise ValueError('Only one output can be added.')

    output_name = name if name else output_tensor.op.name
    self._outputs[output_tensor.name] = explain_metadata.OutputMetadata(
        name=output_name, output_tensor_name=output_tensor.name)

  def get_metadata(self) -> Dict[Text, Any]:
    """Returns the current metadata."""
    current_md = explain_metadata.ExplainMetadata(
        inputs=list(self._inputs.values()),
        outputs=list(self._outputs.values()),
        framework='Tensorflow',
        tags=[constants.METADATA_TAG])
    return current_md.to_dict()

  def _build_input_signature(self, md_entries, graph):
    """Builds an input signature dictionary from input metadata entries."""
    return {md.name: graph.get_tensor_by_name(md.input_tensor_name)
            for md in md_entries.values()}

  def _build_output_signature(self, md_entries, graph):
    """Builds an input signature dictionary from input metadata entries."""
    return {md.name: graph.get_tensor_by_name(md.output_tensor_name)
            for md in md_entries.values()}

  def save_model_with_metadata(self, file_path: Text):
    """Saves the model and the generated metadata to the given file path.

    Args:
      file_path: Path to save the model and the metadata. It can be a GCS bucket
        or a local folder. The folder needs to be empty.

    Returns:
      Full file path where the model and the metadata are written.
    """
    md_dict = self.get_metadata()

    if not self._serving_inputs:
      self._serving_inputs = self._build_input_signature(
          self._inputs, self._session.graph)
    if not self._serving_outputs:
      self._serving_outputs = self._build_output_signature(
          self._outputs, self._session.graph)

    utils.save_graph_model(self._session, file_path, self._serving_inputs,
                           self._serving_outputs, self._tags,
                           **self._saved_model_args)

    common_utils.write_metadata_to_file(md_dict, file_path)
    return file_path
