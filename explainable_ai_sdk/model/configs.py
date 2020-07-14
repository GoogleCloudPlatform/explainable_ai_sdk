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

"""Config classes for explanation methods.
"""
import abc
import dataclasses


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
