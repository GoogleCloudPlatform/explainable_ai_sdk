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

"""Utility functions for models.
"""
import collections

from explainable_ai_sdk.common import explain_metadata
from explainable_ai_sdk.model import constants


def get_modality_input_list_map(
    explain_md):
  """Gets a mapping between modality and input lists.

  Args:
    explain_md: ExplainMetadata object to collect modalities from.

  Returns:
     A dictionary that maps modality to a list of input names.
  """
  modality_input_list_map = collections.defaultdict(list)

  for input_metadata in explain_md.inputs:
    input_name = input_metadata.name
    input_modality = input_metadata.modality
    input_encoding = input_metadata.encoding
    if input_modality in explain_metadata.Modality.values():
      if (input_encoding == explain_metadata.Encoding.BAG_OF_FEATURES or
          input_encoding == explain_metadata.Encoding.BAG_OF_FEATURES_SPARSE):
        for feature_name in input_metadata.index_feature_mapping:
          modality_input_list_map[explain_metadata.Modality.CATEGORICAL].append(
              feature_name)
          modality_input_list_map[constants.ALL_MODALITY].append(feature_name)
          modality_input_list_map[constants.TABULAR_MODALITY].append(
              feature_name)
      else:
        if (input_modality in (explain_metadata.Modality.NUMERIC,
                               explain_metadata.Modality.CATEGORICAL)):
          modality_input_list_map[constants.TABULAR_MODALITY].append(input_name)
        modality_input_list_map[input_modality].append(input_name)
        modality_input_list_map[constants.ALL_MODALITY].append(input_name)
  return modality_input_list_map
