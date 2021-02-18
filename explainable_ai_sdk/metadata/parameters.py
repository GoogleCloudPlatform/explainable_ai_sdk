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

"""Typed objects for metadata parameters.

These classes closely follow parameters in explain_metadata.py. However, this is
the actual user interface. Classes in this module should be used instead of the
ones in explain_metadata.py.
"""
import enum
from typing import Optional, Dict, Any, Union
import dataclasses


@enum.unique
class OverlayType(enum.Enum):
  """Various methods of overlaying the visualization on an input image.

  Possible options:
    NONE: No overlay.
    OVERLAY_ON_ORIGINAL_IMAGE: The attributions are shown on top of the original
      image. The green channel is used for positive attributions and red channel
      is used for the negative attributions. The PolarityType is used to select
      which type of attributions to visualize - positive, negative or both.
    OVERLAY_ON_GRAYSCALE_IMAGE: The attributions are shown on top of grayscaled
      version of the original image. The green channel is used for positive
      attributions and red channel is used for the negative attributions. The
      PolarityType is used to select which type of attributions to visualize -
      positive, negative or both.
    MASK_BLACK: The attributions are used as a mask to reveal predictive parts
      of the image and hide the un-predictive parts. The opacity of the pixels
      in the original image correspond to the intensity of the attributions for
      the corresponding pixel.
  """
  NONE = "none"
  OVERLAY_ON_ORIGINAL_IMAGE = "original"
  OVERLAY_ON_GRAYSCALE_IMAGE = "grayscale"
  MASK_BLACK = "mask"


@enum.unique
class VisualizationType(enum.Enum):
  """Types of the visualization for attributions.

  Possible options:
    PIXELS: Attributions are highlighted via pixels.
    OUTLINES: Attributions are visualized with outlines.
  """
  PIXELS = "pixels"
  OUTLINES = "outlines"


@enum.unique
class ColorMap(enum.Enum):
  """Color maps for creating the attributions heatmap.

  Possible options:
    RED_GREEN: Pink for negative, green for positive attributions.
    PINK_GREEN: Red for negative, green for positive attributions.
    VIRIDIS: Viridis color map.
  """
  RED_GREEN = "red_green"
  PINK_GREEN = "pink_green"
  VIRIDIS = "viridis"


@enum.unique
class Polarity(enum.Enum):
  """Polarity values for highlighting the attributions.

  Possible options:
    POSITIVE: Shows only positive attributions.
    NEGATIVE: Shows only negative attributions.
    BOTH: Shows both positive and negative attributions.
  """
  POSITIVE = "positive"
  NEGATIVE = "negative"
  BOTH = "both"


@dataclasses.dataclass(frozen=True)
class VisualizationParameters(object):
  """Common attributes for visualization parameters.

  It is recommended that all values are specified. If a subset is provided,
  other values will be filled with defaults depending on the attribution method.

  Attributes:
    type: Type of the visualizations. Must take one of VisualizationType values.
    polarity: Whether to only highlight pixels with positive contributions,
      negative contributions, or both.
    color_map: Which color map to use for visualizing attributions.
    clip_above_percentile: Attributions above this percentile will be
      ignored and considered as outliers. Must be in range [0, 100].
    clip_below_percentile: Attributions below this percentile will be
      ignored and considered as outliers. Must be in range [0, 100] and less
      than CLIP_ABOVE_PERCENTILE.
    overlay_type: How to overlay the visualized attributions over input image.
      Must take one of OverlayType values.
    overlay_multiplier: A multiplier in range [0, 1] indicating the fraction
      of the input image to include in the overlayed visualization (the other
      fraction, i.e. 1 - overlay_multiplier, applies to the attributions).
  """
  type: Optional[VisualizationType] = None
  polarity: Optional[Polarity] = None

  color_map: Optional[ColorMap] = None

  clip_above_percentile: Optional[float] = None
  clip_below_percentile: Optional[float] = None

  overlay_type: Optional[OverlayType] = None
  overlay_multiplier: Optional[float] = None

  def asdict(self) -> Dict[str, Union[float, str]]:
    """Returns the dictionary representation of visualization parameters."""
    stripped_dict = _strip_none(self.__dict__)
    return {
        k: v.value if isinstance(v, enum.Enum) else v
        for k, v in stripped_dict.items()
    }


@dataclasses.dataclass(frozen=True)
class DomainInfo(object):
  """Domain of an input (feature).

  Attributes:
    min: The minimum permissible value for this feature.
    max: The maximum permissible value for this feature.
    original_mean: If this input feature has been normalized to a mean
      value of 0, the original_mean specifies the mean value of the
      domain prior to normalization.
    original_stddev: If this input feature has been normalized to a
      standard deviation of 1.0, the original_stddev specifies the
      standard deviation of the domain prior to normalization.
  """
  min: float
  max: float
  original_mean: Optional[float] = None
  original_stddev: Optional[float] = None

  def asdict(self) -> Dict[str, float]:
    """Returns the dictionary representation of the domain info."""
    return _strip_none(self.__dict__)


def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
  """Strips the given dictionary of None values."""
  return {k: v for k, v in d.items() if v is not None}
