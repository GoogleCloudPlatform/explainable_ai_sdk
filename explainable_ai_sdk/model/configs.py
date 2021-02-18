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

"""Config classes for explanation methods.
"""
import abc
from typing import List, Union, Optional
import dataclasses
from explainable_ai_sdk.common import types


class AttributionConfig(abc.ABC):
  """Abstract base class for attribution configs."""


@dataclasses.dataclass(frozen=True)
class IntegratedGradientsConfig(AttributionConfig):
  """Configuration class to hold Integrated Gradients method parameters."""
  step_count: int = 50


@dataclasses.dataclass(frozen=True)
class SampledShapleyConfig(AttributionConfig):
  """Configuration class to hold Sampled Shapley method parameters."""
  path_count: int = 10


class XraiConfig(IntegratedGradientsConfig):
  """Configuration class to hold XRAI parameters."""


@dataclasses.dataclass(frozen=True)
class AttributionParameters:
  """Class to hold attribution parameters.

  If some parameters are left out or set to None, provided values in the
  attribution config and the metadata will be used.

  Attributes:
    label_indices: List of list of integers specifying which output indices to
      be explained for each example in the batch. Outer list represents each
      instance in the provided batch. Inner list represents label indices to be
      explained for each instance. Alternatively, a single list can be provided
      to explain the same indices the whole batch. Note that the number of
      labels to explain for each example should be the same. If unset or set to
      None, top k classess will be explained (specified by top_k property).
    top_k: Integer to specify k in top k highest output indices to explain.
    baselines: A list of baselines for each input. Each baseline needs to be in
      the same format as the input instance being fed to the explain function.
      Baselines can also be provided as an instance for explained signature (if
      the explained signature is different than the serving signature).
      Attributions will be averaged for all baselines.
    attribution_config: Attribution config to override attribution method
      parameters that were provided previously. Note that the class of this
      config cannot be different from what was provided before.
  """
  label_indices: Optional[List[Union[int, List[int]]]] = None
  top_k: Optional[int] = None
  baselines: Optional[List[types.Instance]] = None
  attribution_config: Optional[AttributionConfig] = None
