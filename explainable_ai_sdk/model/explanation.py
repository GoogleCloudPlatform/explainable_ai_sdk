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


"""Attribution classes for holding attributions from explainer services.

The class is a key class for SDK funtionalities (e.g., visualization).
"""
import base64
import io


from matplotlib import image as mpimg
from matplotlib import pyplot as plt

import numpy as np

from explainable_ai_sdk.common import attribution
from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.common import types
from explainable_ai_sdk.model import constants

APPROX_ERROR_THRESHOLD = 0.05


class Explanation(object):
  """Base class for storing explanations."""

  def __init__(self,
               instance_attribution,
               instance,
               modality_input_list_map):
    """Creates an Explanation object.

    Args:
      instance_attribution: Attribution (LabelIndexToAttribution object) for the
        provided instance.
      instance: A dictionary of values representing the data point.
      modality_input_list_map: Dictionary mapping from modality to a list of
        input names.
    """
    self._modality_input_list_map = modality_input_list_map
    self._instance = instance
    self._label_index_to_attributions = instance_attribution

  @classmethod
  def from_ai_platform_response(
      cls, attribution_dict,
      instance,
      modality_input_list_map):
    """Forms an Explanation object from AI Platform explain server response.

    Args:
      attribution_dict: Attribution response from AI Platform.
      instance: A dictionary of values representing the data point.
      modality_input_list_map: Dictionary mapping from modality to a list of
        input names.

    Returns:
      A newly-created Explanation object.
    """
    label_idx_to_attr = attribution.LabelIndexToAttribution.from_list(
        attribution_dict['attributions_by_label'])
    return cls(label_idx_to_attr, instance, modality_input_list_map)

  def get_attribution(self, label_index = None
                     ):
    """Returns an object of the attributions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, return the attribution of the label with highest
        prediction score.

    Returns:
      The attribution object of a specific label.
    """
    if label_index is None:
      label_index = self.get_top_k_indices(1)[0]

    target_label_attr = self._label_index_to_attributions[label_index]
    return target_label_attr

  def get_top_k_indices(self, k=None):
    """Returns top k label indices for the given instance in sorted order."""
    return self._label_index_to_attributions.get_top_k_label_index_list(k)

  def feature_importance(
      self,
      label_index = None,
      modality = constants.ALL_MODALITY):
    """Returns a dict of each feature and the corresponding attribution value.

    If the feature is not 1D (e.g., RGB channels, embeddings), the value is a
    sum of all dimensions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/tabular/all). Note: tabular modality is
        considered to be numeric or categorical.

    Returns:
      A dictionary of features and corresponding feature attribution values
    """
    target_label_attr = self.get_attribution(label_index)
    input_names = self._modality_input_list_map[modality]
    return target_label_attr.feature_importance(input_names)

  def as_tensors(
      self,
      label_index = None,
      modality = constants.ALL_MODALITY):
    """Returns a dict of each feature and the corresponding raw attributions.

    Unlike the feature_importance method, this method does not aggregate the
    attributions, it keeps the attributions in the shape of their original
    dimensions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/tabular/all). Note: tabular modality is
        considered to be numeric or categorical.

    Returns:
      A dictionary of features and corresponding raw feature attribution values
    """
    target_label_attr = self.get_attribution(label_index)
    input_names = self._modality_input_list_map[modality]
    return target_label_attr.as_tensors(input_names)

  def _print_basic_info(
      self,
      label_index = None,
      print_label_index = True):
    """Prints basic information of a specific label.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, return the attribution of the label with highest
        prediction score.
      print_label_index: If true, print label_index information. This can be
        used to turn off label_index output in regression models.
    """
    target_label_attr = self.get_attribution(label_index)

    if print_label_index and (target_label_attr.label_index is not None) and (
        target_label_attr.label_index != -1):
      print('Label Index %d' % target_label_attr.label_index)
    print('Example Score: %.4f' % target_label_attr.example_score)
    print('Baseline Score: %.4f' % target_label_attr.baseline_score)
    if target_label_attr.approx_error is not None:
      print('Approximation Error: %.4f' % target_label_attr.approx_error)

      if target_label_attr.approx_error > APPROX_ERROR_THRESHOLD:
        print('Warning: Approximation error exceeds 5%.')

  def visualize_top_k_features(self,
                               k = 10,
                               label_index = None,
                               modality = constants.ALL_MODALITY):
    """Visualizes attributions.

    Args:
      k: If k is given, visualize top k features. If it is not given, set k=10
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/tabular/all). Note: tabular modality is
        considered to be numeric or categorical.
    """
    importance_dict = self.feature_importance(label_index, modality)
    sorted_feature_names = sorted(
        importance_dict, key=importance_dict.get, reverse=True)

    # Slice to only top k
    sorted_feature_names = sorted_feature_names[:k]

    sorted_attr_values = [importance_dict[key] for key in sorted_feature_names]

    num_features = len(sorted_feature_names)

    if num_features > 0:
      x_pos = list(range(num_features))
      plt.barh(x_pos, sorted_attr_values)
      plt.yticks(x_pos, sorted_feature_names)
      plt.title('Feature attributions')
      plt.ylabel('Feature names')
      plt.xlabel('Attribution value')
      plt.show()

  def _visualize_image_attributions(self, label_index = None):
    """Visualizes image attributions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
    """
    if (explain_metadata.Modality.IMAGE not in self._modality_input_list_map or
        not self._modality_input_list_map[explain_metadata.Modality.IMAGE]):
      return
    target_label_attr = self.get_attribution(label_index)
    input_names = self._modality_input_list_map[explain_metadata.Modality.IMAGE]
    num_features = len(input_names)

    if num_features > 0:
      for input_name in input_names:
        attr_values = target_label_attr.post_processed_attributions
        b64str = attr_values[input_name]['b64_jpeg']
        i = base64.b64decode(b64str)
        i = io.BytesIO(i)
        i = mpimg.imread(i, format='JPG')

        plt.imshow(i, interpolation='nearest')
        plt.show()

  def _visualize_tabular_attributions(self, label_index = None):
    """Visualizes tabular attributions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
    """
    if constants.TABULAR_MODALITY in self._modality_input_list_map:
      self.visualize_top_k_features(
          k=len(self._modality_input_list_map[constants.TABULAR_MODALITY]),
          label_index=label_index,
          modality=constants.TABULAR_MODALITY)

  def visualize_attributions(
      self,
      label_index = None,
      print_label_index = True):
    """Visualizes all types of attributions.

    Args:
      label_index: If label_index is given, return the attribution of the given
        label. If not, will use the label with highest prediction score.
      print_label_index: If true, print label_index information. This can be
        used to turn off label_index output in regression models.
    """
    # Print basic information
    self._print_basic_info(label_index, print_label_index)
    self._visualize_tabular_attributions(label_index)
    self._visualize_image_attributions(label_index)
