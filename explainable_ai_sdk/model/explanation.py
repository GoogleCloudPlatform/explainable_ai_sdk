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

  def get_attribution(self, class_index = None
                     ):
    """Returns an object of the attributions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, return the attribution of the class with highest
        prediction score.

    Returns:
      The attribution object of a specific class.
    """
    if class_index is None:
      label_idx_to_attr = self._label_index_to_attributions
      top_cls_list = label_idx_to_attr.get_top_k_class_index_list()
      class_index = top_cls_list[0]

    target_class_attr = self._label_index_to_attributions[class_index]
    return target_class_attr

  def feature_importance(
      self,
      class_index = None,
      modality = constants.ALL_MODALITY):
    """Returns a dict of each feature and the corresponding attribution value.

    If the feature is not 1D (e.g., RGB channels, embeddings), the value is a
    sum of all dimensions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/all).

    Returns:
      A dictionary of features and corresponding feature attribution values
    """
    target_class_attr = self.get_attribution(class_index)
    input_names = self._modality_input_list_map[modality]
    return target_class_attr.feature_importance(input_names)

  def as_tensors(
      self,
      class_index = None,
      modality = constants.ALL_MODALITY):
    """Returns a dict of each feature and the corresponding raw attributions.

    Unlike the feature_importance method, this method does not aggregate the
    attributions, it keeps the attributions in the shape of their original
    dimensions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/all).

    Returns:
      A dictionary of features and corresponding raw feature attribution values
    """
    target_class_attr = self.get_attribution(class_index)
    input_names = self._modality_input_list_map[modality]
    return target_class_attr.as_tensors(input_names)

  def _print_basic_info(self, class_index = None):
    """Prints basic information of a specific class.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, return the attribution of the class with highest
        prediction score.
    """
    target_class_attr = self.get_attribution(class_index)

    print('Label Index %d' % target_class_attr.label_index)
    print('Example Score: %.4f' % target_class_attr.example_score)
    print('Baseline Score: %.4f' % target_class_attr.baseline_score)
    if target_class_attr.approx_error is not None:
      print('Approximation Error: %.4f' % target_class_attr.approx_error)

      if target_class_attr.approx_error > APPROX_ERROR_THRESHOLD:
        print('Warning: Approximation error exceeds 5%.')

  def visualize_top_k_features(self,
                               k = 10,
                               class_index = None,
                               modality = constants.ALL_MODALITY):
    """Visualizes attributions.

    Args:
      k: If k is given, visualize top k features. If it is not given, set k=10
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
      modality: Tensor modalities to be considered
        (numeric/image/categorical/text/all).
    """
    importance_dict = self.feature_importance(class_index, modality)
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

  def _visualize_image_attributions(self, class_index = None):
    """Visualizes image attributions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
    """
    if (explain_metadata.Modality.IMAGE not in self._modality_input_list_map or
        not self._modality_input_list_map[explain_metadata.Modality.IMAGE]):
      return
    target_class_attr = self.get_attribution(class_index)
    input_names = self._modality_input_list_map[explain_metadata.Modality.IMAGE]
    num_features = len(input_names)

    if num_features > 0:
      for input_name in input_names:
        attr_values = target_class_attr.post_processed_attributions
        b64str = attr_values[input_name]['b64_jpeg']
        i = base64.b64decode(b64str)
        i = io.BytesIO(i)
        i = mpimg.imread(i, format='JPG')

        plt.imshow(i, interpolation='nearest')
        plt.show()

  def _visualize_tabular_attributions(self, class_index = None):
    """Visualizes tabular attributions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
    """
    if (explain_metadata.Modality.CATEGORICAL in self._modality_input_list_map
        and
        self._modality_input_list_map[explain_metadata.Modality.CATEGORICAL]):
      self.visualize_top_k_features(
          k=len(self._modality_input_list_map[
              explain_metadata.Modality.CATEGORICAL]),
          class_index=class_index,
          modality=explain_metadata.Modality.CATEGORICAL)

    if (explain_metadata.Modality.NUMERIC in self._modality_input_list_map and
        self._modality_input_list_map[explain_metadata.Modality.NUMERIC]):
      self.visualize_top_k_features(
          k=len(
              self._modality_input_list_map[explain_metadata.Modality.NUMERIC]),
          class_index=class_index,
          modality=explain_metadata.Modality.NUMERIC)

  def visualize_attributions(self, class_index = None):
    """Visualizes all types of attributions.

    Args:
      class_index: If class_index is given, return the attribution of the given
        class. If not, will use the class with highest prediction score.
    """
    # Print basic information
    self._print_basic_info(class_index)
    self._visualize_tabular_attributions(class_index)
    self._visualize_image_attributions(class_index)
