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

"""Metadata builder for TF1 saved models.

This builder loads a TF1 saved model and creates a metadata builder from the
signature inputs and outputs. It also inherits from GraphMetadataBuilder so that
more inputs can be added/removed.
"""
from typing import Dict, Optional, List
import tensorflow.compat.v1 as tf
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.metadata import utils
from explainable_ai_sdk.metadata.tf.v1 import graph_metadata_builder


class SavedModelMetadataBuilder(graph_metadata_builder.GraphMetadataBuilder):
  """Metadata builder class that accepts a TF1 saved model."""

  def __init__(
      self,
      model_path: str,
      tags: Optional[List[str]] = None,
      signature_name: str = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      outputs_to_explain: Optional[List[str]] = None,
      **kwargs) -> None:
    """Initializes a SavedModelMetadataBuilder object.

    Args:
      model_path: Path to load the saved model from.
      tags: Tags to identify the model graph. If None or empty, TensorFlow's
        default serving tag will be used.
      signature_name: Name of the signature to be explained. Inputs and outputs
        of this signature will be written in the metadata. If not provided, the
        default TensorFlow serving signature will be used.
      outputs_to_explain: List of output names to explain. Only single output is
        supported for now. Hence, the list should contain one element. This
        parameter is required if the model signature (provided via
        signature_name) specifies multiple outputs.
      **kwargs: Any keyword arguments to be passed to SavedModelBuilder's
        add_meta_graph_and_variables function.
    """
    if not tags:
      self._tags = [tf.saved_model.tag_constants.SERVING]
    else:
      self._tags = tags
    self._saved_model_args = kwargs
    # create a graph and session to load the model into.
    self._graph = tf.Graph()
    with self.graph.as_default():
      self._session = tf.Session(graph=self.graph)
      self._metagraph_def = tf.saved_model.loader.load(
          sess=self.session, tags=self._tags, export_dir=model_path)

      if signature_name not in self._metagraph_def.signature_def:
        raise ValueError(
            f"Serving sigdef key {signature_name} not in "
            "the signature def.")
      serving_sigdef = self._metagraph_def.signature_def[signature_name]
    # get tensors for inputs and the outputs of a given signature.
    if outputs_to_explain:
      if len(outputs_to_explain) > 1:
        raise ValueError("Only one output is supported at the moment. "
                         f"Received: {outputs_to_explain}.")
      self._output_to_explain = next(iter(outputs_to_explain))
    else:
      if len(serving_sigdef.outputs) > 1:
        raise ValueError('The signature contains multiple outputs. Specify '
                         'an output via "outputs_to_explain" parameter.')
      self._output_to_explain = next(iter(serving_sigdef.outputs.keys()))
    self._inputs = _create_input_metadata_from_signature(serving_sigdef.inputs)
    self._outputs = _create_output_metadata_from_signature(
        serving_sigdef.outputs, self._output_to_explain)

  @property
  def graph(self) -> tf.Graph:
    return self._graph

  @property
  def session(self) -> tf.Session:
    return self._session

  def save_model_with_metadata(self, file_path: str) -> str:
    """Saves the model and the generated metadata to the given file path.

    Note that this function resaves the loaded model. If the model is not fully
    loaded, then some aspects of the model might be missing. If only the
    metadata is required, use "save_metadata" function.

    Args:
      file_path: Path to save the model and the metadata. It can be a GCS bucket
        or a local folder. The folder needs to be empty.

    Returns:
      Full path where the model and the metadata are written.
    """
    builder = tf.saved_model.builder.SavedModelBuilder(file_path)
    builder.add_meta_graph_and_variables(
        self.session,
        self._tags,
        signature_def_map=self._metagraph_def.signature_def,
        **self._saved_model_args)
    builder.save()
    self.save_metadata(file_path)
    return file_path

  def save_metadata(self, file_path: str) -> None:
    """Saves the metadata to the given folder."""
    utils.write_metadata_to_file(self.get_metadata(), file_path)


def _create_input_metadata_from_signature(
    signature_inputs: Dict[str, tf.Tensor]
    ) -> Dict[str, explain_metadata.InputMetadata]:
  """Creates InputMetadata from signature inputs."""
  return {key: explain_metadata.InputMetadata(key, tensor.name)
          for key, tensor in signature_inputs.items()}


def _create_output_metadata_from_signature(
    signature_outputs: Dict[str, tf.Tensor],
    output_to_explain: Optional[str] = None
    ) -> Dict[str, explain_metadata.OutputMetadata]:
  """Creates OutputMetadata from signature outputs."""
  return {key: explain_metadata.OutputMetadata(key, tensor.name)
          for key, tensor in signature_outputs.items()
          if not output_to_explain or output_to_explain == key}
