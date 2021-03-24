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

"""Definitions for explain metadata and helper functions to read from file."""

import collections
import copy
import json
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
import tensorflow as tf

from explainable_ai_sdk.common import constants
from explainable_ai_sdk.common import semver
from explainable_ai_sdk.common import types
from explainable_ai_sdk.common import utils
from explainable_ai_sdk.common import validation_utils


class Modality(utils.FieldKeys):
  """Modality of the InputMetadata.

  Attributes:
    NUMERIC: Numeric values, scalar or vector.
    IMAGE: Image.
    TEXT: str.
    STRUCTURED: Structured data organized with keyed values.
    CATEGORICAL: Values that are part of categorical features - this could be
      any data type that can be compared for equality, e.g. string or array
      of strings.
  """
  NUMERIC = "numeric"
  IMAGE = "image"
  TEXT = "text"
  STRUCTURED = "structured"
  CATEGORICAL = "categorical"


class FeatureDomain(utils.FieldKeys):
  """Encodes information for domains of feature values.

  Attributes:
    MIN: The minimum permissible value for this feature.
    MAX: The maximum permissible value for this feature.
    ORIGINAL_MEAN: If this input feature has been normalized to a mean
      value of 0, the original_mean specifies the mean value of the
      domain prior to normalization.
    ORIGINAL_STDDEV: If this input feature has been normalized to a
      standard deviation of 1.0, the original_stddev specifies the
      standard deviation of the domain prior to normalization.
  """
  MIN = "min"
  MAX = "max"
  ORIGINAL_MEAN = "original_mean"
  ORIGINAL_STDDEV = "original_stddev"


class Encoding(utils.FieldKeys):
  """How features are encoded into input tensor.

  Attributes:
    IDENTITY: The tensor represents one feature.
    BAG_OF_FEATURES: The tensor represents a bag of features where each index
      maps to a feature.
      eg.
      input [27, 6.0, 150]
      index_feature_mapping ["age", "height", "weight"]
    BAG_OF_FEATURES_SPARSE: The tensor represents a bag of features where each
      index maps to a feature. Zero values in the tensor indicates feature being
      non-existent.
      eg.
      input [2, 0, 5, 0, 1]
      index_feature_mapping ["a", "b", "c", "d", "e"]
    INDICATOR: The tensor is binary where 1 indicates the existence of a feature
      where each index maps to a feature.
      eg.
      input [1, 0, 1, 0, 1]
      index_feature_mapping ["a", "b", "c", "d", "e"]
    COMBINED_EMBEDDING: The input tensor is encoded into a 1D array. Would
      require the encoded tensor. The input tensor is required to have the same
      shape as the encoded tensor.
      eg.
      input ["This", "is", "a", "test", "."]
      encoded [0.1, 0.2, 0.3, 0.4, 0.5]
    CONCAT_EMBEDDING: The input tensor is encoded into a 1D array. Would require
      the encoded tensor. The input tensor is required to have the same shape
      as the encoded tensor.
      eg.
      input ["This", "is", "a", "test", "."]
      encoded [[0.1, 0.2, 0.3, 0.4, 0.5],
               [0.2, 0.1, 0.4, 0.3, 0.5],
               [0.5, 0.1, 0.3, 0.5, 0.4],
               [0.5, 0.3, 0.1, 0.2, 0.4],
               [0.4, 0.3, 0.2, 0.5, 0.1]]
  """
  IDENTITY = "identity"
  BAG_OF_FEATURES = "bag_of_features"
  BAG_OF_FEATURES_SPARSE = "bag_of_features_sparse"
  INDICATOR = "indicator"
  COMBINED_EMBEDDING = "combined_embedding"
  CONCAT_EMBEDDING = "concat_embedding"


class Framework(utils.FieldKeys):
  """Frameworks currently supported for our library.

  Currently, there is no version in frameworks.

  Attributes:
    TENSORFLOW: TensorFlow.
    TENSORFLOW2: TensorFlow2
    XGBOOST: XGBoost.
    CUSTOM_CONTAINER: custom container.
  """
  TENSORFLOW = "tensorflow"
  TENSORFLOW2 = "tensorflow2"
  XGBOOST = "xgboost"
  CUSTOM_CONTAINER = "custom_container"


class MetadataKeys(utils.FieldKeys):
  """Top level keys of Metadata.

  Attributes:
    INPUTS: Key of a list of InputMetadata.
    OUTPUTS: Key of a list of OutputMetadata.
    EMBEDDINGS: Key of a list of EmbeddingMetadata.
    SIGDEF_INPUTS: Key of a list of SigDefInputMetadata.
    PREPARER_VERSION: Key of the semantic version.
    FRAMEWORK: Key of the framework.
    TAGS: A list of tags to annotate the metadata.
  """
  INPUTS = "inputs"
  OUTPUTS = "outputs"
  EMBEDDINGS = "embeddings"
  SIGDEF_INPUTS = "sigdef_inputs"
  SERVING_SIGDEF_KEY = "serving_sigdef_key"
  PREPARER_VERSION = "preparer_version"
  FRAMEWORK = "framework"
  TAGS = "tags"


class InputMetadataKeys(utils.FieldKeys):
  """Keys input dictionary holds and args of InputMetadata.

  Attributes:
    INPUT_TENSOR_NAME: Name of the input tensor.
    INPUT_TENSOR_DTYPE: Dtype of the input tensor.
    DOMAIN: Represents the domain information for this input.
    INDICES_TENSOR_NAME: Specifies the index of the values of the input tensor.
      Required when the input_tensor is a sparse representation.
    INDICES_TENSOR_DTYPE: Dtype of the indices tensor.
    DENSE_SHAPE_TENSOR_NAME: Specifies the dense shape of the input if the input
      is a sparse representation.
    DENSE_SHAPE_TENSOR_DTYPE: Dtype of the dense shape tensor.
    ENCODING: How the features are encoded to the input tensor.
      (One of explain_metadata.Encoding)
    INPUT_BASELINES: A list of baselines used for attribution explanation.
      The shape of each baseline should match the shape of the input tensor.
      If a scalar is provided, we broadcast to the same shape as the input
      tensor.
    INDEX_FEATURE_MAPPING: A list of feature names for each index in the input
      tensor.
      Required when the input encoding is BAG_OF_FEATURES,
      BAG_OF_FEATURES_SPARSE, INDICATOR.
    ENCODED_TENSOR_NAME: Name of encoded tensor.
      Encoded tensor - A transformation of the input tensor.
      e.g. input tensor -> LookupTable -> encoded tensor.
      AttributionExplainer operates on this tensor if provided.
    ENCODED_TENSOR_DTYPE: Dtype of encoded tensor.
    ENCODED_BASELINES: A list of baselines used for attribution explanation.
      The shape of each baseline should match the shape of the encoded tensor.
      If a scalar is provided, we broadcast to the same shape as the
      encoded tensor.
    MODALITY: Modality of the InputMetadata.
      (One of explain_metadata.Modality)
    GRADIENT_TENSOR_NAMES: A dictionary of gradients tensor names with output
      tensor name as key the gradient tensor name as values.
    GRADIENT_TENSOR_DTYPES: A dictionary of gradients tensor dtypes with output
      tensor name as key the gradient tensor dtype as values.
    VISUALIZATION: The type of interpretable visualization to produce from
      attributions, and parameters for producing them. This is an optional
      field.
    WEIGHT_VALUES_NAME: Weight values for inputs. This field is needed for
      multivalent attribution when the input is built with
      tf.weighted_categorical_column. Weights always have sparse representation.
    WEIGHT_VALUES_DTYPE: Dtype of weight values.
    WEIGHT_INDICES_NAME: Indices for the weight values.
    WEIGHT_INDICES_DTYPE: Dtype of weight values.
    WEIGHT_DENSE_SHAPE_NAME: Dense shape of the weight values.
    WEIGHT_DENSE_SHAPE_DTYPE: Dtype of weight dense shape values.
    GROUP_NAME: Name of the group that the input belongs to.
    IS_SEQUENTIAL: Whether this input feature represents a sequential feature.
      Sequential features cannot be evaluated in a batched manner - a batch of
      inputs is considered a sequence of input values for a single prediction.
  """
  VISUALIZATION = "visualization"
  INPUT_TENSOR_NAME = "input_tensor_name"
  INPUT_TENSOR_DTYPE = "input_tensor_dtype"
  DOMAIN = "domain"
  INDICES_TENSOR_NAME = "indices_tensor_name"
  INDICES_TENSOR_DTYPE = "indices_tensor_dtype"
  DENSE_SHAPE_TENSOR_NAME = "dense_shape_tensor_name"
  DENSE_SHAPE_TENSOR_DTYPE = "dense_shape_tensor_dtype"
  ENCODING = "encoding"
  INPUT_BASELINES = "input_baselines"
  INDEX_FEATURE_MAPPING = "index_feature_mapping"
  ENCODED_TENSOR_NAME = "encoded_tensor_name"
  ENCODED_TENSOR_DTYPE = "encoded_tensor_dtype"
  ENCODED_BASELINES = "encoded_baselines"
  MODALITY = "modality"
  GRADIENT_TENSOR_NAMES = "gradient_tensor_names"
  GRADIENT_TENSOR_DTYPES = "gradient_tensor_dtypes"
  WEIGHT_VALUES_NAME = "weight_values_name"
  WEIGHT_VALUES_DTYPE = "weight_values_dtype"
  WEIGHT_INDICES_NAME = "weight_indices_name"
  WEIGHT_INDICES_DTYPE = "weight_indices_dtype"
  WEIGHT_DENSE_SHAPE_NAME = "weight_dense_shape_name"
  WEIGHT_DENSE_SHAPE_DTYPE = "weight_dense_shape_dtype"
  GROUP_NAME = "group_name"
  IS_SEQUENTIAL = "is_sequential"


class DeprecatedInputMetadataKeys(utils.FieldKeys):
  """Deprecated keys from InputMetadataKeys.

  Keeping the keys for backward compat.

  Attributes:
    INPUT_TENSOR_TYPE: Data type of the input tensor.
  """
  INPUT_TENSOR_TYPE = "input_tensor_type"


class OutputMetadataKeys(utils.FieldKeys):
  """Keys output dictionary holds and args of OutputMetadata.

  Attributes:
    OUTPUT_TENSOR_NAME: Name of the output tensor.
    OUTPUT_TENSOR_DTYPE: Dtype of the output tensor.
    INDEX_NAME_MAPPING: A list of label names for each index in the output
      tensor.
    INDEX_NAME_MAPPING_KEY: The key name within the model's output response
      which maps to the class name for the label being explained.
  """
  OUTPUT_TENSOR_NAME = "output_tensor_name"
  OUTPUT_TENSOR_DTYPE = "output_tensor_dtype"
  INDEX_NAME_MAPPING = "index_name_mapping"
  INDEX_NAME_MAPPING_KEY = "index_name_mapping_key"


class EmbeddingMetadataKeys(utils.FieldKeys):
  """Keys embedding dictionary holds and args of EmbeddingMetadata.

  Attributes:
    EMBEDDING_TENSOR_NAME: Name of the embedding tensor.
    EMBEDDING_TENSOR_DTYPE: Dtype of the embedding tensor.
  """
  EMBEDDING_TENSOR_NAME = "embedding_tensor_name"
  EMBEDDING_TENSOR_DTYPE = "embedding_tensor_dtype"


class SigDefInputMetadataKeys(utils.FieldKeys):
  """Keys output dictionary holds and args of SigDefInputMetadata.

  Attributes:
    SIGDEF_INPUT_TENSOR_NAME: Name of the sigdef input tensor.
    SIGDEF_INPUT_TENSOR_DTYPE: DType of the sigdef input tensor.
  """
  SIGDEF_INPUT_TENSOR_NAME = "sigdef_input_tensor_name"
  SIGDEF_INPUT_TENSOR_DTYPE = "sigdef_input_tensor_dtype"


class AttributionVisualizationTypes(utils.FieldKeys):
  """Valid values for the TYPE field in InputMetadata.visualization.

  Attributes:
    PIXELS: pixels.
    OUTLINES: outlines.
  """
  PIXELS = constants.PIXELS
  OUTLINES = constants.OUTLINES


class PixelsVisualizationKeys(utils.FieldKeys):
  """Keys of the post processor that produces pixel visualizations.

  Attributes:
    POLARITY: Whether to only highlight pixels with positive contributions
      ("positive"), negative contributions ("negative"), or both ("both").
    COLOR_MAP: Which color map to use for visualizing attributions.
    BLUR_SIGMA: If set, this specifies the stddev of the gaussian kernel used
      for blurring the visualized attributions.
    CLIP_ABOVE_PERCENTILE: Attributions above this percentile will be
      ignored and considered as outliers. Must be in range [0, 100].
    CLIP_BELOW_PERCENTILE: Attributions below this percentile will be
      ignored and considered as outliers. Must be in range [0, 100] and less
      than CLIP_ABOVE_PERCENTILE.
    OVERLAY_TYPE: How to overlay the visualized attributions over the input
      image. The value associated with this key must map to those defined under
      "OverlayType" below.
    OVERLAY_MULTIPLIER: A multiplier in range [0, 1] indicating the fraction
      of the input image to include in the overlaid visualization (the other
      fraction, i.e. 1 - overlay_multiplier, applies to the attributions).
  """
  # Whether to show positive attributions, negative attributions or both.
  POLARITY = "polarity"

  # Which color map to use for visualizing attributions.
  COLOR_MAP = "color_map"

  # If set, this specifies the stddev of the gaussian kernel used for blurring
  # the visualized attributions.
  BLUR_SIGMA = "blur_sigma"

  # Show attributions within these percentile values.
  CLIP_ABOVE_PERCENTILE = "clip_above_percentile"
  CLIP_BELOW_PERCENTILE = "clip_below_percentile"

  # How to overlay the visualized attributions over the input image.
  OVERLAY_TYPE = "overlay_type"
  OVERLAY_MULTIPLIER = "overlay_multiplier"


class OutlinesVisualizationKeys(utils.FieldKeys):
  """Keys of the post processor that produces outlines-based visualizations.

  Attributes:
    POLARITY: Whether to only highlight pixels with positive contributions
      ("positive"), negative contributions ("negative"), or both ("both").
    COLOR_MAP: Which color map to use for visualizing attributions.
    CLIP_ABOVE_PERCENTILE: Attributions above this percentile will be
      ignored and considered as outliers. Must be in range [0, 100].
    CLIP_BELOW_PERCENTILE: Attributions below this percentile will be
      ignored and considered as outliers. Must be in range [0, 100] and less
      than CLIP_ABOVE_PERCENTILE.
    SHADE_WITHIN_OUTLINES: Whether to shade within the outlines (boolean value,
      default true).
    OVERLAY_TYPE: How to overlay the visualized attributions over input image.
      The value associated with this key must map to those defined under
      "OverlayType" below.
    OVERLAY_MULTIPLIER: A multiplier in range [0, 1] indicating the fraction
      of the input image to include in the overlaid visualization (the other
      fraction, i.e. 1 - overlay_multiplier, applies to the attributions).
  """
  # Whether to show positive attributions, negative attributions or both.
  POLARITY = "polarity"

  # Which color map to use for visualizing attributions.
  COLOR_MAP = "color_map"

  # Show attributions within these percentile values.
  CLIP_ABOVE_PERCENTILE = "clip_above_percentile"
  CLIP_BELOW_PERCENTILE = "clip_below_percentile"

  # Whether to shade within the outlines (boolean value, default true).
  SHADE_WITHIN_OUTLINES = "shade_within_outlines"

  # How to overlay the visualized attributions over input image.
  OVERLAY_TYPE = "overlay_type"
  OVERLAY_MULTIPLIER = "overlay_multiplier"


class OverlayType(utils.FieldKeys):
  """Various methods of overlaying the visualization on input image."""

  # No overlay.
  NONE = "none"

  # The attributions are shown on top of the original image.
  # The green channel is used for positive attributions and red channel is used
  # for the negative attributions. The PolarityType is used to select which type
  # of attributions to visualize - positive, negative or both.
  OVERLAY_ON_ORIGINAL_IMAGE = "original"

  # The attributions are shown on top of grayscaled version of the original
  # image. The green channel is used for positive attributions and red channel
  # is used for the negative attributions. The PolarityType is used to select
  # which type of attributions to visualize - positive, negative or both.
  OVERLAY_ON_GRAYSCALE_IMAGE = "grayscale"

  # The attributions are used as a mask to reveal predictive parts of the image
  # and hide the un-predictive parts. The opacity of the pixels in the original
  # image correspond to the intensity of the attributions for the corresponding
  # pixel.
  MASK_BLACK = "mask_black"


class AttributionVisualizationRequiredKeys(utils.FieldKeys):
  """Required keys for InputMetadata.visualization (must be set).

  The valid values corresponding to the TYPE key are provided under
  AttributionVisualizationTypes.

  Attributes:
    TYPE: The visualisation type.
  """
  TYPE = "type"


class AttributionVisualizationKeys(object):
  """Valid set of keys for InputMetadata.visualization."""
  REQUIRED_KEYS = AttributionVisualizationRequiredKeys.values()
  KEYS_FOR_TYPE = {
      AttributionVisualizationTypes.PIXELS: PixelsVisualizationKeys,
      AttributionVisualizationTypes.OUTLINES: OutlinesVisualizationKeys
  }

  @classmethod
  def values(cls, visualization_type: str = None):
    """Returns the set of valid keys under InputMetadata.visualization."""
    if not visualization_type:
      return AttributionVisualizationKeys.REQUIRED_KEYS
    return AttributionVisualizationKeys.REQUIRED_KEYS.union(
        AttributionVisualizationKeys.KEYS_FOR_TYPE[visualization_type].values())


class InputMetadata(object):
  """Metadata for holding input information."""

  def __init__(self,
               name: str,
               input_tensor_name: str = None,
               input_tensor_dtype: str = None,
               domain: Dict[str, Any] = None,
               encoding: str = Encoding.IDENTITY,
               indices_tensor_name: str = None,
               indices_tensor_dtype: str = None,
               dense_shape_tensor_name: str = None,
               dense_shape_tensor_dtype: str = None,
               input_baselines: List[Any] = None,
               index_feature_mapping: List[Any] = None,
               encoded_tensor_name: str = None,
               encoded_tensor_dtype: str = None,
               encoded_baselines: List[Any] = None,
               modality: str = Modality.NUMERIC,
               gradient_tensor_names: Dict[str, str] = None,
               gradient_tensor_dtypes: Dict[str, str] = None,
               weight_values_name: str = None,
               weight_values_dtype: str = None,
               weight_indices_name: str = None,
               weight_indices_dtype: str = None,
               weight_dense_shape_name: str = None,
               weight_dense_shape_dtype: str = None,
               visualization: Dict[str, str] = None,
               group_name: str = None,
               is_sequential: bool = False):
    self._name = name
    self._input_tensor_name = input_tensor_name
    self._input_tensor_dtype = input_tensor_dtype
    self._domain = domain
    self._encoding = encoding
    self._indices_tensor_name = indices_tensor_name
    self._indices_tensor_dtype = indices_tensor_dtype
    self._dense_shape_tensor_name = dense_shape_tensor_name
    self._dense_shape_tensor_dtype = dense_shape_tensor_dtype
    self._input_baselines = input_baselines
    self._index_feature_mapping = index_feature_mapping
    self._encoded_tensor_name = encoded_tensor_name
    self._encoded_tensor_dtype = encoded_tensor_dtype
    self._encoded_baselines = encoded_baselines
    self._modality = modality
    self._gradient_tensor_names = gradient_tensor_names
    self._gradient_tensor_dtypes = gradient_tensor_dtypes
    self._weight_values_name = weight_values_name
    self._weight_values_dtype = weight_values_dtype
    self._weight_indices_name = weight_indices_name
    self._weight_indices_dtype = weight_indices_dtype
    self._weight_dense_shape_name = weight_dense_shape_name
    self._weight_dense_shape_dtype = weight_dense_shape_dtype
    self._visualization = visualization
    self._group_name = group_name
    self._is_sequential = is_sequential

    if (self.visualization and
        AttributionVisualizationRequiredKeys.TYPE in self.visualization):
      # making the visualization accept both outlines/pixels and Outlines/Pixels
      viz_type = self.visualization[AttributionVisualizationRequiredKeys.TYPE]
      self.visualization[
          AttributionVisualizationRequiredKeys.TYPE] = viz_type.lower()
    self._validate()
    self._tensor_name_to_dtype_mapping = self._get_tensor_name_to_dtype_mapping(
    )

  def _validate(self):
    """Validates InputExplainMetadata.

    InputExplainMetadata should satisfy the following properties.
    1. Property type validation.
    2. If encoded_baselines is present input_baselines should not be.
    """
    self._validate_type_and_enum()

    self._validate_index_feature_mapping()

    if self._domain is not None:
      self._validate_domain()

    self._validate_tensors()

    self._validate_baselines()

  def _validate_type_and_enum(self):
    validation_utils.validate_object_init_type_hint(self)
    validation_utils.validate_is_in(self._encoding, "encoding",
                                    Encoding.values(), "InputMetadata")
    validation_utils.validate_is_in(self._modality, "modality",
                                    Modality.values(), "InputMetadata")

  def _validate_index_feature_mapping(self):
    """Validates index_feature_mapping related fields in InputExplainMetadata."""
    logging.debug(
        "XAI Validation :: Metadata: [InputMetadata] "
        "`index_feature_mapping` should not contain duplicate features.")
    if self._index_feature_mapping is not None:
      if len(self._index_feature_mapping) != len(
          set(self._index_feature_mapping)):
        raise ValueError("index_feature_mapping contains duplicate features.")
    elif self._encoding in {
        Encoding.BAG_OF_FEATURES, Encoding.BAG_OF_FEATURES_SPARSE,
        Encoding.INDICATOR}:
      raise ValueError(
          "index_feature_mapping must be provided when encoding is {}.".format(
              self._encoding))

  def _validate_tensors(self):
    """Validates tensor related fields in InputExplainMetadata."""
    logging.debug("XAI Validation :: Metadata: [InputMetadata] "
                  "Keys in `gradient_tensor_names` should match keys in "
                  "`gradient_tensor_dtypes`.")
    if self._gradient_tensor_names and self._gradient_tensor_dtypes and set(
        self._gradient_tensor_names.keys()) != set(
            self._gradient_tensor_dtypes.keys()):
      raise ValueError("keys in gradient_tensor_names does not match keys in "
                       "gradient_tensor_dtypes.")
    logging.debug("XAI Validation :: Metadata: [InputMetadata] All weight "
                  "tensors (values, indices, dense_shape) or none should "
                  "present.")
    if not ((self._weight_indices_name is not None and
             self._weight_values_name is not None and
             self._weight_dense_shape_name is not None) or
            (not self._weight_indices_name and not self._weight_values_name and
             not self._weight_dense_shape_name)):
      raise ValueError("You must provide all weight tensors (values, indices, "
                       "dense_shape) or none for the feature {}.".format(
                           self._name))

  def _validate_baselines(self):
    """Validates input_baselines and encoded_baselines."""
    logging.debug("XAI Validation :: Metadata: [InputMetadata] Only one of "
                  "`encoded_baselines` or `input_baselines` should present.")
    if (self._encoded_baselines is not None and
        self._input_baselines is not None):
      raise ValueError("Got both encoded_baselines and input_baselines "
                       "for input_metadata {}. "
                       "Only one should be present.".format(self._name))
    logging.debug(
        "XAI Validation :: Metadata: [InputMetadata] The number of "
        "`encoded_baselines` should be at most %d.",
        constants.MAX_NUM_BASELINES)
    if (self._encoded_baselines is not None and
        len(self._encoded_baselines) > constants.MAX_NUM_BASELINES):
      raise ValueError("Got too many encoded baselines. "
                       "Please limit the number of baselines to {:d}.".format(
                           constants.MAX_NUM_BASELINES))
    logging.debug(
        "XAI Validation :: Metadata: [InputMetadata] The number of "
        "`input_baselines` should be at most %d.", constants.MAX_NUM_BASELINES)
    if (self._input_baselines is not None and
        len(self._input_baselines) > constants.MAX_NUM_BASELINES):
      raise ValueError("Got too many input baselines. "
                       "Please limit the number of baselines to {:d}.".format(
                           constants.MAX_NUM_BASELINES))

  def _validate_domain(self):
    """Validates InputMetadata.DOMAIN config provided by the user."""
    logging.debug(
        "XAI Validation :: Metadata: [InputMetadata] "
        "Feature domain `keys` should be a subset of `%s`.",
        sorted(map(str, set(FeatureDomain.values()))))
    if not set(self._domain.keys()).issubset(set(FeatureDomain.values())):
      raise ValueError(
          "Invalid key '{}' provided for {}.".format
          (set(self._domain.keys()).difference(set(
              FeatureDomain.values())), InputMetadataKeys.DOMAIN))
    self._validate_domain_range()

  def _validate_domain_range(self):
    """Validates feature domain range in the InputMetadata.DOMAIN config."""
    # Checks range values in the feature domain.
    logging.debug("XAI Validation :: Metadata: [InputMetadata] Feature domain:"
                  "Both or none of `min` and `max` should be specified.")
    logging.debug("XAI Validation :: Metadata: [InputMetadata] Feature domain:"
                  "Both `min` and `max` should be floats and `min` < `max`.")
    logging.debug(
        "XAI Validation :: Metadata: [InputMetadata] "
        "Feature domain range should only be specified for inputs "
        "with `%s` encoding.", Encoding.IDENTITY)
    if FeatureDomain.MIN in self._domain or FeatureDomain.MAX in self._domain:
      if (FeatureDomain.MAX not in self._domain or
          FeatureDomain.MIN not in self._domain):
        raise ValueError("Both min and max must be specified; got {}.".format(
            repr(self._domain)))
      if (not isinstance(self._domain[FeatureDomain.MIN], (int, float)) or
          not isinstance(self._domain[FeatureDomain.MAX], (int, float))):
        raise ValueError("Min and max must both be floats. Got {}.".format(
            repr(self._domain)))
      if self._domain[FeatureDomain.MIN] > self._domain[FeatureDomain.MAX]:
        raise ValueError("Feature domain invalid, MAX must be larger than MIN:"
                         " {}.".format(repr(self._domain)))
      if self._encoding != Encoding.IDENTITY:
        raise ValueError("domain range must only be specified for inputs "
                         "with identity encoding (the encoding provided is "
                         "{}).".format(self._encoding))

  @property
  def name(self):
    return self._name

  @property
  def encoded_tensor_name(self):
    return self._encoded_tensor_name

  @property
  def encoded_tensor_dtype(self):
    return self._encoded_tensor_dtype

  @encoded_tensor_dtype.setter
  def encoded_tensor_dtype(self, encoded_tensor_dtype: str):
    validation_utils.validate_is_instance(encoded_tensor_dtype,
                                          "encoded_tensor_dtype", str)
    self._encoded_tensor_dtype = encoded_tensor_dtype

  @property
  def input_tensor_name(self):
    return self._input_tensor_name

  @property
  def input_tensor_dtype(self):
    return self._input_tensor_dtype

  @input_tensor_dtype.setter
  def input_tensor_dtype(self, input_tensor_dtype: str):
    validation_utils.validate_is_instance(input_tensor_dtype,
                                          "input_tensor_dtype", str)
    self._input_tensor_dtype = input_tensor_dtype

  @property
  def indices_tensor_name(self):
    return self._indices_tensor_name

  @property
  def indices_tensor_dtype(self):
    return self._indices_tensor_dtype

  @indices_tensor_dtype.setter
  def indices_tensor_dtype(self, indices_tensor_dtype: str) -> None:
    validation_utils.validate_is_instance(indices_tensor_dtype,
                                          "indices_tensor_dtype", str)
    self._indices_tensor_dtype = indices_tensor_dtype

  @property
  def dense_shape_tensor_name(self):
    return self._dense_shape_tensor_name

  @property
  def dense_shape_tensor_dtype(self):
    return self._dense_shape_tensor_dtype

  @dense_shape_tensor_dtype.setter
  def dense_shape_tensor_dtype(self, dense_shape_tensor_dtype: str) -> None:
    validation_utils.validate_is_instance(dense_shape_tensor_dtype,
                                          "dense_shape_tensor_dtype", str)
    self._dense_shape_tensor_dtype = dense_shape_tensor_dtype

  @property
  def input_baselines(self):
    return self._input_baselines

  @input_baselines.setter
  def input_baselines(self, input_baselines: List[Any]) -> None:
    validation_utils.validate_is_instance(input_baselines, "input_baselines",
                                          list)
    self._input_baselines = input_baselines

  @property
  def encoded_baselines(self):
    return self._encoded_baselines

  @encoded_baselines.setter
  def encoded_baselines(self, encoded_baselines: List[Any]) -> None:
    validation_utils.validate_is_instance(encoded_baselines,
                                          "encoded_baselines", list)
    self._encoded_baselines = encoded_baselines

  @property
  def encoding(self):
    return self._encoding

  @property
  def index_feature_mapping(self):
    return self._index_feature_mapping

  @property
  def modality(self):
    return self._modality

  @modality.setter
  def modality(self, modality: str) -> None:
    validation_utils.validate_is_instance(modality, "modality", str)
    validation_utils.validate_is_in(modality, "modality", Modality.values())
    self._modality = modality

  @property
  def domain(self):
    return self._domain

  @property
  def is_sequential(self):
    return self._is_sequential

  @property
  def gradient_tensor_names(self):
    return self._gradient_tensor_names

  @property
  def gradient_tensor_dtypes(self):
    return self._gradient_tensor_dtypes

  @gradient_tensor_dtypes.setter
  def gradient_tensor_dtypes(self, gradient_tensor_dtypes: Dict[str, str]):
    validation_utils.validate_is_instance(gradient_tensor_dtypes,
                                          "gradient_tensor_dtypes", dict)
    self._gradient_tensor_dtypes = gradient_tensor_dtypes

  @property
  def weight_values_name(self):
    return self._weight_values_name

  @property
  def weight_values_dtype(self):
    return self._weight_values_dtype

  @weight_values_dtype.setter
  def weight_values_dtype(self, weight_values_dtype: str) -> None:
    validation_utils.validate_is_instance(weight_values_dtype,
                                          "weight_values_dtype", str)
    self._weight_values_dtype = weight_values_dtype

  @property
  def weight_indices_name(self):
    return self._weight_indices_name

  @property
  def weight_indices_dtype(self):
    return self._weight_indices_dtype

  @weight_indices_dtype.setter
  def weight_indices_dtype(self, weight_indices_dtype: str) -> None:
    validation_utils.validate_is_instance(weight_indices_dtype,
                                          "weight_indices_dtype", str)
    self._weight_indices_dtype = weight_indices_dtype

  @property
  def weight_dense_shape_name(self):
    return self._weight_dense_shape_name

  @property
  def weight_dense_shape_dtype(self):
    return self._weight_dense_shape_dtype

  @weight_dense_shape_dtype.setter
  def weight_dense_shape_dtype(self, weight_dense_shape_dtype: str) -> None:
    validation_utils.validate_is_instance(weight_dense_shape_dtype,
                                          "weight_dense_shape_dtype", str)
    self._weight_dense_shape_dtype = weight_dense_shape_dtype

  @property
  def group_name(self):
    return self._group_name

  def update_gradient_tensor_names(self,
                                   output_name,
                                   gradient_tensor_name):
    if self._gradient_tensor_names is None:
      self._gradient_tensor_names = {}
    self._gradient_tensor_names[output_name] = gradient_tensor_name

  def gradient_tensor_name(self, output_name=None):
    """Get gradient tensor name by output name in output metadata.

    Args:
      output_name: that the gradient corresponds to. If not provided, fetch
        the first element in the dictionary.

    Returns:
      gradient tensor name.
        (output_tensor_name respect to the input/encoded tensor name.)


    """
    if output_name:
      return self._gradient_tensor_names[output_name]
    else:
      return next(iter(self._gradient_tensor_names.values()))

  def _get_tensor_name_to_dtype_mapping(self) -> Dict[str, str]:
    """Get tensor name to dtype mapping for all tensors in InputMetadata.

    Returns:
      A dictionary with tensor name as key and dtype as value.
    """
    tensor_name_to_dtype_mapping = {}

    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._input_tensor_name, self._input_tensor_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._indices_tensor_name,
                            self._indices_tensor_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._encoded_tensor_name,
                            self._encoded_tensor_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._indices_tensor_name,
                            self._indices_tensor_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._dense_shape_tensor_name,
                            self._dense_shape_tensor_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._weight_values_name, self._weight_values_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._weight_indices_name,
                            self._weight_indices_dtype)
    _update_map_if_not_none(tensor_name_to_dtype_mapping,
                            self._weight_dense_shape_name,
                            self._weight_dense_shape_dtype)
    if (self._gradient_tensor_names is not None and
        self._gradient_tensor_dtypes is not None):
      for k in self._gradient_tensor_names:
        _update_map_if_not_none(tensor_name_to_dtype_mapping,
                                self._gradient_tensor_names[k],
                                self._gradient_tensor_dtypes[k])
    return tensor_name_to_dtype_mapping

  @property
  def tensor_name_to_dtype_mapping(self):
    return self._tensor_name_to_dtype_mapping

  @property
  def visualization(self):
    return self._visualization

  @classmethod
  def from_dict(cls, name: str, in_dict: Dict[str, Any]) -> "InputMetadata":
    """Construct a dense input metadata object from an ordered python dictionary.

    The input metadata looks like the following:
    (Indicator tensor)
    {
      "input_tensor_name": "dense_feat0:0",
      "encoding": "bag_of_features_sparse",
      "input_baselines": [[0, 0, 0], [1, 1, 1], [1, 1, 0]],
      "index_feature_mapping": ["baseketball", "baseball", "football"]
    }

    (Sparse tensor encoded into one single embedding tensor)
    {
      "input_tensor_name": "input:0",
      "indices_tensor_name": "indices:0",
      "dense_shape_tensor_name": "dense_shape:0",
      "encoded_tensor_name": "encoded:0",
      "encoding": "combined_embedding",
      "encoded_baselines": [[0.1, 0.2, 0.6, 0.3, 0.9],
                            [0.2, 0.3, 0.5, 0.2, 0.7],
                            [0.8, 0.9, 1.5, 2.1, 3.7]]
    }

    (Gradient prepared)
    {
      "input_tensor_name": "dense_feat0:0",
      "input_tensor_dtype": "float32",
      "encoding": "indicator",
      "input_baselines": [0, 1, [1, 1, 0]],
      "index_feature_mapping": ["baseketball", "baseball", "football"],
      "gradient_tensor_names": {"output_1": "grad_1:0", "output_2": "grad_2:0"}
      "gradient_tensor_dtypes": {"output_1": "float32", "output_2": "float32"}
    }

    (XGBoost tensor)
    {
      "encoding": "indicator",
      "index_feature_mapping": ["baseketball", "baseball", "football"]
    }

    All fields are optional.

    Args:
      name: Friendly input name.
      in_dict: Python dict containing the dense input metadata.

    Returns:
      InputMetadata object.
    """
    in_dict = collections.OrderedDict([
        (k.lower(), v) for k, v in in_dict.items()
    ])

    keys_delta = set(in_dict.keys()).difference(
        InputMetadataKeys.values().union(DeprecatedInputMetadataKeys.values()))
    if keys_delta:
      raise ValueError(
          "Unexpected input keys {} while parsing explain metadata.".format(
              list(keys_delta)))

    kwargs = {"name": name}
    kwargs.update({
        k: in_dict[k]
        for k in InputMetadataKeys.values()
        if k in in_dict
    })

    return cls(**kwargs)

  def to_dict(self, remove_empty_vals=True) -> Dict[str, Any]:
    """Constructs a dict representation of InputMetadata."""
    input_dict = {
        InputMetadataKeys.VISUALIZATION:
            self.visualization,
        InputMetadataKeys.INPUT_TENSOR_NAME:
            self.input_tensor_name,
        InputMetadataKeys.INPUT_TENSOR_DTYPE:
            self.input_tensor_dtype,
        InputMetadataKeys.ENCODING:
            self.encoding,
        InputMetadataKeys.MODALITY:
            self.modality,
        InputMetadataKeys.DOMAIN:
            self.domain,
        InputMetadataKeys.INDICES_TENSOR_NAME:
            self.indices_tensor_name,
        InputMetadataKeys.INDICES_TENSOR_DTYPE:
            self.indices_tensor_dtype,
        InputMetadataKeys.DENSE_SHAPE_TENSOR_NAME:
            self.dense_shape_tensor_name,
        InputMetadataKeys.DENSE_SHAPE_TENSOR_DTYPE:
            self.dense_shape_tensor_dtype,
        InputMetadataKeys.INPUT_BASELINES:
            self.input_baselines,
        InputMetadataKeys.INDEX_FEATURE_MAPPING:
            self.index_feature_mapping,
        InputMetadataKeys.ENCODED_TENSOR_NAME:
            self.encoded_tensor_name,
        InputMetadataKeys.ENCODED_TENSOR_DTYPE:
            self.encoded_tensor_dtype,
        InputMetadataKeys.ENCODED_BASELINES:
            self.encoded_baselines,
        InputMetadataKeys.GRADIENT_TENSOR_NAMES:
            self.gradient_tensor_names,
        InputMetadataKeys.GRADIENT_TENSOR_DTYPES:
            self.gradient_tensor_dtypes,
        InputMetadataKeys.WEIGHT_VALUES_NAME:
            self.weight_values_name,
        InputMetadataKeys.WEIGHT_VALUES_DTYPE:
            self.weight_values_dtype,
        InputMetadataKeys.WEIGHT_INDICES_NAME:
            self.weight_indices_name,
        InputMetadataKeys.WEIGHT_INDICES_DTYPE:
            self.weight_indices_dtype,
        InputMetadataKeys.WEIGHT_DENSE_SHAPE_NAME:
            self.weight_dense_shape_name,
        InputMetadataKeys.WEIGHT_DENSE_SHAPE_DTYPE:
            self.weight_dense_shape_dtype,
        InputMetadataKeys.GROUP_NAME:
            self.group_name
    }
    if self.is_sequential:
      input_dict[InputMetadataKeys.IS_SEQUENTIAL] = True
    if remove_empty_vals:
      input_dict = utils.remove_empty_vals(input_dict)
    return input_dict


class OutputMetadata(object):
  """Metadata for holding a model's output information."""

  def __init__(self,
               name: str,
               output_tensor_name: Optional[str] = None,
               output_tensor_dtype: Optional[str] = None,
               index_name_mapping: Optional[List[Any]] = None,
               index_name_mapping_key: Optional[str] = None):
    """Initializes an OutputMetadata object.

    Args:
      name: Friendly output name.
      output_tensor_name: Output tensor name.
      output_tensor_dtype: Dtype tensor name.
      index_name_mapping: Maps each index in the output array to a label name.
      index_name_mapping_key: The key name within the model's output response
        which maps to the class name for the label being explained.
    """
    self._name = name
    self._output_tensor_name = output_tensor_name
    self._output_tensor_dtype = output_tensor_dtype
    self._index_name_mapping = index_name_mapping
    self._index_name_mapping_key = index_name_mapping_key
    self._validate()
    self._tensor_name_to_dtype_mapping = (
        self._get_tensor_name_to_dtype_mapping())

  def _validate(self):
    """Validate OutputMetadata."""
    validation_utils.validate_object_init_type_hint(self)
    logging.debug("XAI Validation :: Metadata: [OutputMetadata] "
                  "`index_feature_mapping` classes should be unique.")
    logging.debug("XAI Validation :: Metadata: [OutputMetadata] "
                  "Only one of `index_class_mapping` or "
                  "`index_name_mapping_key` should present.")
    if self._index_name_mapping is not None:
      if len(self._index_name_mapping) != len(set(self._index_name_mapping)):
        raise ValueError("index_feature_mapping contains duplicate classes.")
      if self._index_name_mapping_key is not None:
        raise ValueError(
            "Only one of index_class_mapping or index_name_mapping_key "
            "must be set at a time.")

  @property
  def name(self):
    return self._name

  @property
  def output_tensor_name(self):
    return self._output_tensor_name

  @property
  def output_tensor_dtype(self):
    return self._output_tensor_dtype

  @output_tensor_dtype.setter
  def output_tensor_dtype(self, output_tensor_dtype: str) -> None:
    validation_utils.validate_is_instance(output_tensor_dtype,
                                          "output_tensor_dtype", str)
    self._output_tensor_dtype = output_tensor_dtype

  @property
  def index_name_mapping(self):
    return self._index_name_mapping

  @property
  def index_name_mapping_key(self):
    return self._index_name_mapping_key

  def _get_tensor_name_to_dtype_mapping(self) -> Dict[str, str]:
    """Get tensor name to dtype mapping for the tensor in OutputMetadata.

    Returns:
      A dictionary with tensor name as key and dtype as value.
    """
    tensor_name_to_dtype_mapping = {}
    if self._output_tensor_name and self._output_tensor_dtype:
      tensor_name_to_dtype_mapping.update(
          {self._output_tensor_name: self._output_tensor_dtype})
    return tensor_name_to_dtype_mapping

  @property
  def tensor_name_to_dtype_mapping(self):
    return self._tensor_name_to_dtype_mapping

  @classmethod
  def from_dict(cls, name: str, out_dict: Dict[str, Any]) -> "OutputMetadata":
    """Construct output metadata object from ordered python dictionary.

    The output metadata json looks like the following:
    {
      "output_tensor_name": "prob:0",
      "index_name_mapping": ["horse", "zebra", "elk"]
    }

    All fields are optional.

    Args:
      name: Output metadata name.
      out_dict: Python dict containing the output metadata.

    Returns:
      OutputMetadata object.
    """
    out_dict = collections.OrderedDict([
        (k.lower(), v) for k, v in out_dict.items()
    ])
    keys_delta = set(out_dict.keys()).difference(
        OutputMetadataKeys.values())
    if keys_delta:
      raise ValueError(
          "Unexpected output keys {} while parsing explain metadata.".format(
              list(keys_delta)))

    kwargs = {"name": name}
    kwargs.update({
        k: out_dict[k]
        for k in OutputMetadataKeys.values()
        if k in out_dict
    })

    return cls(**kwargs)

  def to_dict(self, remove_empty_vals=True) -> Dict[str, Any]:
    """Constructs a dict representation of OutputMetadata."""
    output_dict = {
        OutputMetadataKeys.OUTPUT_TENSOR_NAME: self.output_tensor_name,
        OutputMetadataKeys.OUTPUT_TENSOR_DTYPE: self.output_tensor_dtype,
        OutputMetadataKeys.INDEX_NAME_MAPPING: self.index_name_mapping,
        OutputMetadataKeys.INDEX_NAME_MAPPING_KEY: self.index_name_mapping_key,
    }
    if remove_empty_vals:
      output_dict = utils.remove_empty_vals(output_dict)
    return output_dict


class EmbeddingMetadata(object):
  """Metadata for holding a model's embedding information.

  Attributes:
      name: Friendly embedding name.
      embedding_tensor_name: Embedding tensor name.
      embedding_tensor_dtype: Dtype tensor name.
      tensor_name_to_dtype_mapping: Mapping from tensor name to dtype.
  """

  def __init__(self,
               name: str,
               embedding_tensor_name: str,
               embedding_tensor_dtype: Optional[str] = None):
    """Initializes an EmbeddingMetadata object.

    Args:
      name: Friendly embedding name.
      embedding_tensor_name: Embedding tensor name.
      embedding_tensor_dtype: Dtype tensor name.
    """
    self._name = name
    self._embedding_tensor_name = embedding_tensor_name
    self._embedding_tensor_dtype = embedding_tensor_dtype
    self._validate()
    self._tensor_name_to_dtype_mapping = (
        self._get_tensor_name_to_dtype_mapping())

  def _validate(self):
    """Validate EmbeddingMetadata."""
    validation_utils.validate_object_init_type_hint(self)

  @property
  def name(self):
    return self._name

  @property
  def embedding_tensor_name(self):
    return self._embedding_tensor_name

  @property
  def embedding_tensor_dtype(self):
    return self._embedding_tensor_dtype

  @embedding_tensor_dtype.setter
  def embedding_tensor_dtype(self, embedding_tensor_dtype: str) -> None:
    validation_utils.validate_is_instance(embedding_tensor_dtype,
                                          "embedding_tensor_dtype", str)
    self._embedding_tensor_dtype = embedding_tensor_dtype

  def _get_tensor_name_to_dtype_mapping(self) -> Dict[str, str]:
    """Get tensor name to dtype mapping for the tensor in EmbeddingMetadata.

    Returns:
      A dictionary with tensor name as key and dtype as value.
    """
    tensor_name_to_dtype_mapping = {}
    if self._embedding_tensor_name and self._embedding_tensor_dtype:
      tensor_name_to_dtype_mapping.update(
          {self._embedding_tensor_name: self._embedding_tensor_dtype})
    return tensor_name_to_dtype_mapping

  @property
  def tensor_name_to_dtype_mapping(self):
    return self._tensor_name_to_dtype_mapping

  @classmethod
  def from_dict(cls, name: str,
                emb_dict: Dict[str, Any]) -> "EmbeddingMetadata":
    """Construct an embedding metadata object from a python dictionary.

    The embedding metadata json looks like the following:
    {
      "embedding_tensor_name": "prob:0",
    }

    All fields are optional.

    Args:
      name: Embedding metadata name.
      emb_dict: Python dict containing the embedding metadata.

    Returns:
      EmbeddingMetadata object.
    """
    emb_dict = collections.OrderedDict(
        (k.lower(), v) for k, v in emb_dict.items()
    )
    keys_delta = set(emb_dict.keys()).difference(EmbeddingMetadataKeys.values())
    if keys_delta:
      raise ValueError(
          "Unexpected embedding keys {} while parsing explain metadata.".format(
              list(keys_delta)))

    kwargs = {"name": name}
    kwargs.update({
        k: emb_dict[k]
        for k in EmbeddingMetadataKeys.values()
        if k in emb_dict
    })

    return cls(**kwargs)

  def to_dict(self, remove_empty_vals=True) -> Dict[str, Any]:
    """Constructs a dict representation of EmbeddingMetadata."""
    embedding_dict = {
        EmbeddingMetadataKeys.EMBEDDING_TENSOR_NAME:
            self.embedding_tensor_name,
        EmbeddingMetadataKeys.EMBEDDING_TENSOR_DTYPE:
            self.embedding_tensor_dtype,
    }
    if remove_empty_vals:
      embedding_dict = utils.remove_empty_vals(embedding_dict)
    return embedding_dict


class SigDefInputMetadata(object):
  """Metadata for holding a model's sigdef input information.

  SigDef inputs are derived from the base user input to the sigdef and are, by
  definition, not meant to be explained so they don't appear in InputMetadata in
  the explain metadata.
  For example, for image models, the sigdef input is typically a base64 encoded
  binary blob which get decoded in the sigdef and the decoded 4D tensor is what
  ultimately is explained and therefore appears in InputMetadata.
  SigDef inputs are used when input itself cannot fulfill the sigdef.

  SigDefInputMetadata is not intended to be set by the user. The model preparer
  will be responsible to set it.
  """

  def __init__(self,
               name: str,
               sigdef_input_tensor_name: Optional[str] = None,
               sigdef_input_tensor_dtype: Optional[str] = None):
    """Initializes an SigDefInputMetadata object.

    Args:
      name: Friendly output name.
      sigdef_input_tensor_name: SigDef input tensor name.
      sigdef_input_tensor_dtype: SigDef input tensor dtype.
    """
    self._name = name
    self._sigdef_input_tensor_name = sigdef_input_tensor_name
    self._sigdef_input_tensor_dtype = sigdef_input_tensor_dtype
    self._validate()
    self._tensor_name_to_dtype_mapping = self._get_tensor_name_to_dtype_mapping(
    )

  def _validate(self):
    """Validate SigDefInputMetadata."""
    validation_utils.validate_object_init_type_hint(self)

  @property
  def name(self):
    return self._name

  @property
  def sigdef_input_tensor_name(self):
    return self._sigdef_input_tensor_name

  @property
  def sigdef_input_tensor_dtype(self):
    return self._sigdef_input_tensor_dtype

  @sigdef_input_tensor_dtype.setter
  def sigdef_input_tensor_dtype(self, sigdef_input_tensor_dtype: str) -> None:
    validation_utils.validate_is_instance(sigdef_input_tensor_dtype,
                                          "sigdef_input_tensor_dtype", str)
    self._sigdef_input_tensor_dtype = sigdef_input_tensor_dtype

  def _get_tensor_name_to_dtype_mapping(self) -> Dict[str, str]:
    """Get tensor name to dtype mapping for all tensors in SigDefInputMetadata.

    Returns:
      A dictionary with tensor name as key and dtype as value.
    """
    tensor_name_to_dtype_mapping = {}
    if self._sigdef_input_tensor_name and self._sigdef_input_tensor_dtype:
      tensor_name_to_dtype_mapping.update(
          {self._sigdef_input_tensor_name: self._sigdef_input_tensor_dtype})

    if self._name and self._sigdef_input_tensor_dtype:
      tensor_name_to_dtype_mapping.update(
          {self._name: self._sigdef_input_tensor_dtype})
    return tensor_name_to_dtype_mapping

  @property
  def tensor_name_to_dtype_mapping(self):
    return self._tensor_name_to_dtype_mapping

  @classmethod
  def from_dict(cls, name: str,
                sigdef_in_dict: Dict[str, Any]) -> "SigDefInputMetadata":
    """Construct sigdef_input metadata object from ordered python dictionary.

    The sigdef_input metadata json looks like the following:
    {
      "sigdef_input_tensor_name": "example:0",
    }

    All fields are optional.

    Args:
      name: Output metadata name.
      sigdef_in_dict: dict containing the sigdef_input metadata.

    Returns:
      SigDefInputMetadata object.
    """
    sigdef_in_dict = collections.OrderedDict([
        (k.lower(), v) for k, v in sigdef_in_dict.items()
    ])
    keys_delta = set(sigdef_in_dict.keys()).difference(
        SigDefInputMetadataKeys.values())
    if keys_delta:
      raise ValueError("Unexpected sigdef_input keys ['{}'] while parsing "
                       "explain metadata.".format(",".join(keys_delta)))

    kwargs = {"name": name}
    kwargs.update({
        k: sigdef_in_dict[k]
        for k in SigDefInputMetadataKeys.values()
        if k in sigdef_in_dict
    })

    return cls(**kwargs)

  def to_dict(self, remove_empty_vals=True) -> Dict[str, Any]:
    """Constructs a dict representation of SigDefInputMetadata."""
    sigdef_in_dict = {
        SigDefInputMetadataKeys.SIGDEF_INPUT_TENSOR_NAME:
            self.sigdef_input_tensor_name,
        SigDefInputMetadataKeys.SIGDEF_INPUT_TENSOR_DTYPE:
            self.sigdef_input_tensor_dtype,
    }
    if remove_empty_vals:
      sigdef_in_dict = utils.remove_empty_vals(sigdef_in_dict)
    return sigdef_in_dict


class ExplainMetadata(object):
  """Class representing explain metadata."""

  def __init__(
      self, inputs: Optional[List[InputMetadata]] = None,
      outputs: Optional[List[OutputMetadata]] = None,
      framework: Optional[str] = Framework.TENSORFLOW,
      sigdef_inputs: Optional[List[SigDefInputMetadata]] = None,
      preparer_version: Optional[semver.SemanticVersion] = None,
      embeddings: Optional[List[EmbeddingMetadata]] = None,
      tags: Optional[List[str]] = None,
      serving_sigdef_key: Optional[str] = None):
    """Creates a ExplainMetadata instance.

    Args:
      inputs: List of input metadata.
      outputs: List of output metadata.
      framework: ML framework for the model.
      sigdef_inputs: List of sigdef_input metadata.
      preparer_version: Semantic version of the preparer.
      embeddings: List of embedding metadata.
      tags: List of tags to annotate the metadata.
      serving_sigdef_key: key for serving signature graph. Only used in
        TF models.
    """
    self._inputs = inputs
    self._outputs = outputs
    self._embeddings = embeddings
    self._preparer_version = preparer_version
    self._framework = framework
    self._tags = tags
    self._sigdef_inputs = [] if sigdef_inputs is None else sigdef_inputs
    self._serving_sigdef_key = serving_sigdef_key

    self._input_tensor_names = self._get_input_tensor_names()
    self._indices_tensor_names = self._get_indices_tensor_names()
    self._dense_shape_tensor_names = self._get_dense_shape_tensor_names()
    self._encoded_tensor_names = self._get_encoded_tensor_names()
    self._sparse_tensor_names = self._get_sparse_tensor_names()
    self._dense_tensor_names = self._get_dense_tensor_names()
    self._sparse_input_dense_tensor_names = (
        self._get_sparse_input_dense_tensor_names())
    self._gradient_tensor_names = self._get_gradient_tensor_names()
    self._input_baselines = self._get_input_baselines()
    self._encoded_baselines = self._get_encoded_baselines()
    self._output_tensor_names = self._get_output_tensor_names()
    self._output_names = self._get_output_names()
    self._weight_tensor_names = self._get_weight_tensor_names()
    self._all_sparse_tensor_names = self._get_all_sparse_tensor_names()
    self._embedding_tensor_names = self._get_embedding_tensor_names()

    self._sigdef_input_tensor_names = self._get_sigdef_input_tensor_names()

    self._inputs_name_mapping = self._get_inputs_name_mapping()
    self._outputs_name_mapping = self._get_outputs_name_mapping()
    self._outputs_index_name_mapping = self._get_outputs_index_name_mapping()
    self._explained_tensor_names = self._get_explained_tensor_names()

    # Dictionary of all the tensor names to dtype mapping.
    self._tensor_name_to_dtype_mapping = self._get_tensor_name_to_dtype_mapping(
        )

    # Create an inverse _input_tensor_names dictionary.
    self._input_names = {v: k for k, v in self._input_tensor_names.items()}

    self._validate()

  @property
  def inputs_name_mapping(self):
    return self._inputs_name_mapping

  def _get_inputs_name_mapping(self):
    """Build input name mapping dict for fast retrieval by name."""
    inputs_name_mapping = {}
    if self.inputs:
      for input_md in self.inputs:
        if input_md.name in inputs_name_mapping:
          raise ValueError(
              "Duplicate input name '{}' found.".format(input_md.name))
        inputs_name_mapping[input_md.name] = input_md
    return inputs_name_mapping

  @property
  def outputs_name_mapping(self):
    return self._outputs_name_mapping

  def _get_outputs_name_mapping(self):
    """Build output name mapping dict for fast retrieval by name."""
    outputs_name_mapping = {}
    if self.outputs:
      for output_md in self.outputs:
        if output_md.name in outputs_name_mapping:
          raise ValueError(
              "Duplicate output name '{}' found.".format(output_md.name))
        outputs_name_mapping[output_md.name] = output_md
    return outputs_name_mapping

  @property
  def outputs_index_name_mapping(self):
    return self._outputs_index_name_mapping

  def _get_outputs_index_name_mapping(self):
    """Builds a mapping dict from output name to index name mapping."""
    if not self.outputs:
      return {}
    outputs_index_name_mapping = {}
    for output_md in self.outputs:
      if output_md.index_name_mapping:
        outputs_index_name_mapping[output_md.name] = (
            output_md.index_name_mapping)
        # Avoid generating a None key for metadata w/o output_tensor_name.
        if output_md.output_tensor_name:
          outputs_index_name_mapping[output_md.output_tensor_name] = (
              output_md.index_name_mapping)
    return outputs_index_name_mapping

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def embeddings(self):
    return self._embeddings

  @property
  def preparer_version(self):
    return self._preparer_version

  @preparer_version.setter
  def preparer_version(self, preparer_version: semver.SemanticVersion):
    validation_utils.validate_is_instance(preparer_version, "preparer_version",
                                          semver.SemanticVersion)
    self._preparer_version = preparer_version

  @property
  def framework(self):
    return self._framework

  @property
  def tags(self):
    return self._tags

  @property
  def sigdef_inputs(self):
    return self._sigdef_inputs

  @property
  def serving_sigdef_key(self):
    return self._serving_sigdef_key

  def update_serving_sigdef_key(self, serving_key):
    self._serving_sigdef_key = serving_key

  def append_sigdef_input(self, sigdef_input: SigDefInputMetadata):
    """Append SigDefInputMetadata to ExplainMetadata."""
    if not isinstance(sigdef_input, SigDefInputMetadata):
      raise TypeError("Cannot append sigdef input of type {}.".format(
          type(sigdef_input)))
    self._sigdef_inputs.append(sigdef_input)

  def input_by_name(self, name):
    return self._inputs_name_mapping.get(name, None)

  def output_by_name(self, name):
    return self._outputs_name_mapping.get(name, None)

  def embedding_by_name(self, name):
    return self._embeddings_name_mapping.get(name, None)

  @property
  def input_tensor_names(self) -> Dict[str, str]:
    return self._input_tensor_names

  def _get_input_tensor_names(self) -> Dict[str, str]:
    """Returns all input tensor names in inputs with metadata name as keys."""
    input_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.input_tensor_name:
          input_tensor_names[in_md.name] = in_md.input_tensor_name
        elif self.framework not in (Framework.TENSORFLOW,
                                    Framework.TENSORFLOW2):
          # If tensor name is not provided, it means that the model
          # can accept md name directly for evaluations. We'll set
          # tensor names to be the md name so that rest of XAI lib
          # can  continue with the convention of using tensor names
          # as keys. Only set the name when Framework is not Tensorflow
          input_tensor_names[in_md.name] = in_md.name
    return input_tensor_names

  @property
  def indices_tensor_names(self) -> Dict[str, str]:
    return self._indices_tensor_names

  def _get_indices_tensor_names(self) -> Dict[str, str]:
    """Returns all indices tensor names in inputs with metadata name as keys."""
    indices_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.indices_tensor_name:
          indices_tensor_names[in_md.name] = in_md.indices_tensor_name
    return indices_tensor_names

  @property
  def dense_shape_tensor_names(self) -> Dict[str, str]:
    return self._dense_shape_tensor_names

  def _get_dense_shape_tensor_names(self) -> Dict[str, str]:
    """Returns dense_shape tensor names in inputs with metadata name as keys."""
    dense_shape_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.dense_shape_tensor_name:
          dense_shape_tensor_names[in_md.name] = in_md.dense_shape_tensor_name
    return dense_shape_tensor_names

  @property
  def sparse_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    return self._sparse_tensor_names

  def _get_sparse_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    """Returns tensor names of TF sparse tensors.

    Returns:
      A dictionary mapping from name to trio of tensors composing a TF sparse
      tensors.
    """
    sparse_tensors = {}
    logging.debug(
        "XAI Validation :: Metadata: [ExplainMetadata] "
        "Both `indices_tensor_name` and "
        "`dense_shape_tensor_name` should present/absent for input tensor.")
    for name in self.input_tensor_names:
      if (name in self.indices_tensor_names) != (
          name in self.dense_shape_tensor_names):
        raise ValueError(
            "both indices_tensor_name and dense_shape_tensor_name must be "
            "present/absent for {}.".format(name))
      if name in self.indices_tensor_names:
        sparse_tensors[name] = types.SparseTensorNames(
            self.input_tensor_names[name], self.indices_tensor_names[name],
            self.dense_shape_tensor_names[name])
    return sparse_tensors

  @property
  def sparse_input_dense_tensor_names(self) -> Dict[str, str]:
    return self._sparse_input_dense_tensor_names

  def _get_sparse_input_dense_tensor_names(self) -> Dict[str, str]:
    """Returns input tensor names if they are encoded."""
    tensors = {}
    for name in self.input_tensor_names:
      if (name not in self.indices_tensor_names and
          name in self.encoded_tensor_names):
        tensors[name] = self.input_tensor_names[name]
    return tensors

  @property
  def dense_tensor_names(self) -> Dict[str, str]:
    return self._dense_tensor_names

  def _get_dense_tensor_names(self) -> Dict[str, str]:
    """Returns dense tensor names in a dictionary from name to tensor name."""
    if self._inputs is None:
      return {}
    names = {}
    for in_md in self._inputs:
      if in_md.encoded_tensor_name is not None:
        names[in_md.name] = in_md.encoded_tensor_name
      elif in_md.input_tensor_name is not None:
        if in_md.indices_tensor_name is None:
          names[in_md.name] = in_md.input_tensor_name
      else:
        names[in_md.name] = in_md.name

    return names

  @property
  def encoded_tensor_names(self) -> Dict[str, str]:
    return self._encoded_tensor_names

  def _get_encoded_tensor_names(self) -> Dict[str, str]:
    """Returns all encoded tensor names in inputs with metadata name as keys."""
    encoded_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.encoded_tensor_name:
          encoded_tensor_names[in_md.name] = in_md.encoded_tensor_name
    return encoded_tensor_names

  @property
  def gradient_tensor_names(self) -> Dict[str, str]:
    return self._gradient_tensor_names

  def gradient_tensors_for_output(self, output_name: str) -> Dict[str, str]:
    """Returns gradient tensor names of all inputs for given output.

    Args:
      output_name: Name of the output tensor, with respect to which
        gradients are desired.

    Returns:
      Fetch dict of gradient tensors to fetch for given output_name.
    """
    if not output_name:
      raise ValueError("output_name must be provided.")
    if not self._inputs:
      return {}
    gradient_tensor_names = {}
    for in_md in self._inputs:
      if not in_md.gradient_tensor_names:
        # We're going to return gradients for inputs where it's set, and
        # ignore inputs where it's not set.
        continue
      if output_name not in in_md.gradient_tensor_names:
        # Raise an exception instead of ignoring this in_md because
        # the caller explicitly asked for the given output_name.
        raise ValueError(
            'gradient tensor for output "{}" w.r.t. input "{}" not set.'.format
            (output_name, in_md.name))
      gradient_tensor_name = in_md.gradient_tensor_names[output_name]
      gradient_tensor_names[gradient_tensor_name] = gradient_tensor_name
    return gradient_tensor_names

  @property
  def weight_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    return self._weight_tensor_names

  def _get_weight_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    """Returns from input name to weight sparse tensors."""
    weight_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.weight_values_name:
          weight_tensor_names[in_md.name] = types.SparseTensorNames(
              in_md.weight_values_name, in_md.weight_indices_name,
              in_md.weight_dense_shape_name)
    return weight_tensor_names

  @property
  def all_sparse_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    return self._all_sparse_tensor_names

  def _get_all_sparse_tensor_names(self) -> Dict[str, types.SparseTensorNames]:
    """Returns a dictionary containing all TF sparse tensor triplets.

    Returns:
      A dictionary mapping from friendly names to sparse tensor triplets. If
      input metadata contains weight tensors, they are added to the dictionary
      keyed by friendly name and '__weights' suffix.
    """
    all_sparse_tensors = copy.deepcopy(self.sparse_tensor_names)
    if self.weight_tensor_names:
      all_sparse_tensors = utils.merge_dict(
          all_sparse_tensors,
          utils.suffix_dict_keys(self.weight_tensor_names, "__weights"))
    return all_sparse_tensors

  @property
  def explained_tensor_names(self) -> Dict[str, str]:
    return self._explained_tensor_names

  def _get_explained_tensor_names(self) -> Dict[str, str]:
    """Returns the explained tensor names with tensor name as keys.

    Explained tensors refer to fully resolved values of model inputs that can be
    operated on by the explainer.

    The model graph can be considered to be composed of 3 overlapping
    sub-graphs:
      (1) model inputs --> explained tensors (encoded dense tensors,
                                              sparse values, sparse indices,
                                              sparse shape, and weight tensors)
      (2) explained tensors --> model outputs
      (3) explained tensors, label index --> gradients, model outputs
    On this model, (1) produces the data that explainers will operate on.
    Explainers use (2) and (3) to operate on the data and generate attributions.
    """

    dense_inputs = utils.merge_dict(self.input_tensor_names,
                                    self.encoded_tensor_names)
    input_dense_tensors = collections.OrderedDict()
    for _, value in dense_inputs.items():
      input_dense_tensors[value] = value
    input_sparse_tensors = collections.OrderedDict()
    # If encoded tensors exist, input tensor can be thought of as sparse tensor
    # even for fixed length (no indices tensor).
    for key in self.encoded_tensor_names:
      input_md = self.input_by_name(key)
      if input_md is None:
        raise ValueError(
            "Invalid metadata. Input missing for named input {}.".format(key))
      _update_map_if_not_none(input_sparse_tensors, input_md.input_tensor_name,
                              input_md.input_tensor_name)
    input_weight_sparse_tensors = collections.OrderedDict()
    for key in self.indices_tensor_names:
      input_md = self.input_by_name(key)
      if input_md is None:
        raise ValueError(
            "Invalid metadata. Input missing for named input {}.".format(key))
      _update_map_if_not_none(input_sparse_tensors, input_md.input_tensor_name,
                              input_md.input_tensor_name)
      _update_map_if_not_none(input_sparse_tensors,
                              input_md.indices_tensor_name,
                              input_md.indices_tensor_name)
      _update_map_if_not_none(input_sparse_tensors,
                              input_md.dense_shape_tensor_name,
                              input_md.dense_shape_tensor_name)
      _update_map_if_not_none(input_weight_sparse_tensors,
                              input_md.weight_values_name,
                              input_md.weight_values_name)
      _update_map_if_not_none(input_weight_sparse_tensors,
                              input_md.weight_indices_name,
                              input_md.weight_indices_name)
      _update_map_if_not_none(input_weight_sparse_tensors,
                              input_md.weight_dense_shape_name,
                              input_md.weight_dense_shape_name)

    return utils.merge_dict(
        input_dense_tensors,
        utils.merge_dict(input_weight_sparse_tensors, input_sparse_tensors))

  def _get_gradient_tensor_names(self):
    """Returns all gradient tensor names in inputs with metadata name as keys.
    """
    gradient_tensor_names = {}
    if self._inputs:
      for in_md in self._inputs:
        if in_md.gradient_tensor_names:
          gradient_tensor_names[in_md.name] = copy.deepcopy(
              in_md.gradient_tensor_names)
    return gradient_tensor_names

  @property
  def input_baselines(self):
    """Returns a list of baselines where each baseline is a dictionary.

    Returns:
      A list of dictionaries where the keys are the input_metadata name and the
      values being the numpy array baseline of input tensors.

    For example:
    [{'input_1': [0, 0, 0], 'input_2': ['pad', 'pad'], 'input_3': -.5},
     {'input_1': [2, 2, 2], 'input_2': ['', ''], 'input_3': .3},
     {'input_1': [1, 1, 1], 'input_2': ['oov', 'oov'], 'input_3': .1},
    ]
    """
    return self._input_baselines

  def _get_input_baselines(self):
    """Returns a list of baselines where each baseline is a dictionary."""
    input_baselines = []
    if self._inputs:
      for in_md in self._inputs:
        if in_md.input_baselines:
          self._append_baselines(in_md.input_baselines,
                                 in_md.name,
                                 input_baselines)
    return input_baselines

  @property
  def encoded_baselines(self):
    """Returns a list of baselines where each baseline is a dictionary.

    Returns:
      A list of dictionaries where the keys are the input_metadata name and the
      values being the numpy array baseline of encoded tensors.

    For example:
    [{'encoded_1': [0, 0, 0], 'encoded_2': [[.3], [.8]], 'encoded_3': -.5},
     {'encoded_1': [2, 2, 2], 'encoded_2': [[.2], [.1]], 'encoded_3': .3},
     {'encoded_1': [1, 1, 1], 'encoded_2': [[.7], [.9]], 'encoded_3': .1},
    ]
    """
    return self._encoded_baselines

  def _get_encoded_baselines(self):
    """Returns a list of baselines where each baseline is a dictionary."""
    encoded_baselines = []
    if self._inputs:
      for in_md in self._inputs:
        if in_md.encoded_baselines:
          self._append_baselines(in_md.encoded_baselines,
                                 in_md.name,
                                 encoded_baselines)
    return encoded_baselines

  def _append_baselines(self,
                        source_baselines: List[Any],
                        source_name: str,
                        baselines: List[Dict[str, Any]]):
    """Add each baseline in a list to a dictionary in a list of the same index.

    Args:
      source_baselines: A list of baselines to be added.
      source_name: Name of the source of the baselines.
      baselines: A list of dictionaries to collect the baselines from different
        source.
    """
    for i, baseline in enumerate(source_baselines):
      if i >= len(baselines):
        baselines.append({})
      baselines[i][source_name] = baseline

  @property
  def output_tensor_names(self) -> Dict[str, str]:
    return self._output_tensor_names

  def _get_output_tensor_names(self) -> Dict[str, str]:
    """Returns all output tensor names in inputs with metadata name as keys."""
    output_tensor_names = {}
    if self._outputs:
      for out_md in self._outputs:
        if out_md.output_tensor_name:
          output_tensor_names[out_md.name] = out_md.output_tensor_name
        else:
          # Custom container does not have tensor name so will keep friendly
          # name as value.
          output_tensor_names[out_md.name] = out_md.name
    return output_tensor_names

  @property
  def embedding_tensor_names(self) -> Dict[str, str]:
    return self._embedding_tensor_names

  def _get_embedding_tensor_names(self) -> Dict[str, str]:
    """Returns all embedding tensor names in embeddings with metadata name as keys."""
    if not self._embeddings:
      return {}
    return {emb_md.name: emb_md.embedding_tensor_name
            for emb_md in self._embeddings if emb_md.embedding_tensor_name}

  @property
  def output_names(self) -> Dict[str, str]:
    return self._output_names

  @property
  def input_names(self) -> Dict[str, str]:
    return self._input_names

  def _get_output_names(self) -> Dict[str, str]:
    """Returns all output metadata names in inputs with tensor name as keys."""
    output_names = {}
    if self._outputs:
      for out_md in self._outputs:
        if out_md.output_tensor_name:
          output_names[out_md.output_tensor_name] = out_md.name
    return output_names

  @property
  def sigdef_input_tensor_names(self):
    return self._sigdef_input_tensor_names

  def _get_sigdef_input_tensor_names(self):
    """Returns all tensor names in sigdef_inputs with metadata name as keys."""
    sigdef_input_tensor_names = {}
    if self._sigdef_inputs:
      for sigdef_in_md in self._sigdef_inputs:
        if sigdef_in_md.sigdef_input_tensor_name:
          sigdef_input_tensor_names[
              sigdef_in_md.name] = sigdef_in_md.sigdef_input_tensor_name
    return sigdef_input_tensor_names

  def _get_tensor_name_to_dtype_mapping(self) -> Dict[str, str]:
    """Get tensor name to dtype mapping for all tensors in ExplainMetadata.

    Returns:
      A dictionary with tensor name as key and dtype as value.
    """
    tensor_name_to_dtype_mapping = {}
    if self._inputs:
      for in_md in self._inputs:
        tensor_name_to_dtype_mapping.update(in_md.tensor_name_to_dtype_mapping)
    if self._outputs:
      for out_md in self._outputs:
        tensor_name_to_dtype_mapping.update(out_md.tensor_name_to_dtype_mapping)
    if self._sigdef_inputs:
      for sigdef_in_md in self._sigdef_inputs:
        tensor_name_to_dtype_mapping.update(
            sigdef_in_md.tensor_name_to_dtype_mapping)
    return tensor_name_to_dtype_mapping

  def _validate(self):
    """Validate common properties of explain metadata.

    Properties to satisfy.
    1. The number of baseline for each input should be the same.
    2. input_tensor_names must exist when framework is tensorflow
    """
    validation_utils.validate_object_init_type_hint(self)
    baseline_counter = collections.Counter()
    for input_baseline in self._input_baselines:
      baseline_counter.update(input_baseline.keys())
    for encoded_baseline in self._encoded_baselines:
      baseline_counter.update(encoded_baseline.keys())
    logging.debug("XAI Validation :: Metadata: [ExplainMetadata] "
                  "All inputs should have the same number of baselines.")
    if len(set(baseline_counter.values())) > 1:
      raise ValueError("Not all inputs have the same number of baselines.")
    logging.debug("XAI Validation :: Metadata: [ExplainMetadata] "
                  "`Input_tensor_names` must exist when `framework` is "
                  "`[tensorflow, tensorflow2]`.")
    if (self._framework in (Framework.TENSORFLOW, Framework.TENSORFLOW2) and
        not self._input_tensor_names):
      raise ValueError("input_tensor_names must exist when framework is "
                       "tensorflow/tensorflow2.")

  def get_dtype_with_tensor_name(self, tensor_name: str) -> Optional[str]:
    if tensor_name in self._tensor_name_to_dtype_mapping:
      return self._tensor_name_to_dtype_mapping[tensor_name]
    return None

  @classmethod
  def from_dict(cls, md_dict: Dict[str, Any]) -> "ExplainMetadata":
    """Construct explain_metadata object from ordered python dictionary.

    Args:
      md_dict: dict representing the explain metadata.

    Returns:
      explain metadata object.
    """
    inputs = []
    outputs = []
    embeddings = []
    preparer_version, framework = None, None
    sigdef_inputs = []
    tags = []
    serving_sigdef_key = None

    for k, v in md_dict.items():
      if k == MetadataKeys.OUTPUTS:
        # Read outputs.
        for out_key in v:
          outputs.append(OutputMetadata.from_dict(out_key, v[out_key]))
      elif k == MetadataKeys.INPUTS:
        # Read inputs.
        for in_key in v:
          inputs.append(InputMetadata.from_dict(in_key, v[in_key]))
      elif k == MetadataKeys.EMBEDDINGS:
        # Read embeddings.
        for em_key in v:
          embeddings.append(EmbeddingMetadata.from_dict(em_key, v[em_key]))
      elif k == MetadataKeys.PREPARER_VERSION:
        preparer_version = semver.SemanticVersion(v)
      elif k == MetadataKeys.FRAMEWORK:
        if v.lower() not in Framework.values():
          raise ValueError("Unsupported or misspelled framework: '{}'."
                           " Supported values: {}.".format(
                               v, Framework.values()))
        framework = v.lower()
      elif k == MetadataKeys.SIGDEF_INPUTS:
        # Read sigdef_inputs:
        for sigdef_in_key in v:
          sigdef_inputs.append(
              SigDefInputMetadata.from_dict(sigdef_in_key, v[sigdef_in_key]))
      elif k == MetadataKeys.TAGS:
        tags = v
      elif k == MetadataKeys.SERVING_SIGDEF_KEY:
        serving_sigdef_key = v
      else:
        raise ValueError(
            "Unexpected top-level key '{}' "
            "while parsing explain metadata.".format(k))

    return cls(
        inputs=inputs,
        outputs=outputs,
        preparer_version=preparer_version,
        sigdef_inputs=sigdef_inputs,
        framework=framework,
        embeddings=embeddings,
        tags=tags,
        serving_sigdef_key=serving_sigdef_key)

  @classmethod
  def from_json(cls, json_str: str) -> "ExplainMetadata":
    """Construct ExplainMetadata from a json string representation.

    The json string will be parsed while preserving the key ordering, this is
    important because the explain metadata has a notion of indexed inputs and
    outputs so the original ordering in json is meaningful.

    Args:
      json_str: json representation of ExplainMetadata.

    Returns:
      A ExplainMetadata Object.
    """
    md_dict = json.loads(json_str, object_pairs_hook=collections.OrderedDict)
    md_dict = collections.OrderedDict([
        (k.lower(), v) for k, v in md_dict.items()
    ])
    return cls.from_dict(md_dict)

  def to_dict(self) -> Dict[str, Any]:
    """Constructs a dictionary representation of ExplainMetadata."""
    output_dict = collections.OrderedDict([
        (o.name, o.to_dict()) for o in self.outputs
    ])
    input_dict = collections.OrderedDict([
        (i.name, i.to_dict()) for i in self.inputs
    ])
    ret_dict = {MetadataKeys.OUTPUTS: output_dict,
                MetadataKeys.INPUTS: input_dict}
    if self.embeddings:
      embedding_dict = {e.name: e.to_dict() for e in self.embeddings}
      ret_dict[MetadataKeys.EMBEDDINGS] = embedding_dict
    if self.preparer_version:
      ret_dict[
          MetadataKeys.PREPARER_VERSION] = self.preparer_version.to_string()
    if self.framework:
      ret_dict[MetadataKeys.FRAMEWORK] = self.framework
    if self.sigdef_inputs:
      sigdef_input_dict = {si.name: si.to_dict() for si in self.sigdef_inputs}
      ret_dict[MetadataKeys.SIGDEF_INPUTS] = sigdef_input_dict
    if self.serving_sigdef_key:
      ret_dict[MetadataKeys.SERVING_SIGDEF_KEY] = self.serving_sigdef_key
    if self.tags:
      ret_dict[MetadataKeys.TAGS] = self.tags
    return ret_dict

  def to_json(self,
              indent: Optional[int] = None,
              separators: Optional[Tuple[str, str]] = None) -> str:
    """Constructs a json representation of ExplainMetadata.

    Args:
      indent: If indent is a non-negative integer, then JSON array elements and
        object members will be pretty-printed with that indent level. An indent
        level of 0 will only insert newlines. None is the most compact
        representation.
      separators: Should be (item_separator, key_separator) tuple. None is the
        default representation.

    Returns:
      Serialized json string.
    """
    return json.dumps(self.to_dict(), indent=indent, separators=separators)

  def to_file(self, filename: str) -> None:
    """Writes the json representation of ExplainMetadata to desired file path."""
    with tf.io.gfile.GFile(filename, "w") as f:
      f.write(self.to_json(indent=4, separators=(",", ": ")))

  @classmethod
  def from_file(cls, filename: str) -> "ExplainMetadata":
    """Reads and parses metadata file.

    Args:
      filename: Full path of the file on gcs such as gs://bucket/folder

    Returns:
      A ExplainMetadata object inferred from the given file.
    """
    with tf.io.gfile.GFile(filename, "r") as f:
      return cls.from_json(f.read())


def _update_map_if_not_none(d, key, val):
  """Updates the given dictionary with (key,val) pair if both are not None."""
  if key is not None and val is not None:
    d.update({key: val})


def explain_metadata_instance_check(explain_md: ExplainMetadata):
  """Checks if the metadata instance is valid."""
  logging.debug("XAI Validation :: Metadata: [ExplainMetadata instance check] "
                "Metadata should not be `None`.")
  if not explain_md:
    raise ValueError("Model must have required metadata.")
  validation_utils.validate_is_instance(explain_md, "Metadata", ExplainMetadata,
                                        "ExplainMetadata instance check")
  logging.debug("XAI Validation :: Metadata: [ExplainMetadata instance check] "
                "Metadata should have at least one input.")
  if not explain_md.inputs:
    raise ValueError("explain_md must have a set of inputs.")
  logging.debug("XAI Validation :: Metadata: [ExplainMetadata instance check] "
                "Metadata should have exactly one output.")
  if not explain_md.outputs:
    raise ValueError("Metadata must have at least one output.")
  if len(explain_md.outputs) > 1:
    raise ValueError("Only one output is supported now.")


def validate_visualization(explain_md: ExplainMetadata):
  """Checks that the visualization property has valid values."""

  for in_md in explain_md.inputs:
    if in_md.visualization is None:
      return  # Visualization config not provided.
    logging.debug(
        "XAI Validation :: Metadata: [IG input visualization config] "
        "`modality` should be `image` when visualization config is provided.")
    if in_md.modality != Modality.IMAGE:
      raise ValueError(
          "Cannot visualize attributions on non-image inputs with "
          "modailty: {} ".format(in_md.modality))
    _validate_visualization_key(in_md)

    #  Overlay type must be in the predefined set
    if PixelsVisualizationKeys.OVERLAY_TYPE in in_md.visualization:
      overlay_type = in_md.visualization[PixelsVisualizationKeys.OVERLAY_TYPE]
      validation_utils.validate_is_in(overlay_type, "overlay_type",
                                      OverlayType.values(),
                                      "IG input visualization config")
    _validate_visualization_clip_range(in_md)


def _validate_visualization_key(in_md: InputMetadata):
  """Checks that the visualization keys are valid."""
  logging.debug(
      "XAI Validation :: Metadata: [IG input visualization config] "
      "Required key(s): `%s` must be specified in visualization config.",
      repr(AttributionVisualizationKeys.values()))
  # Verify that all required keys are set.
  required_keys_not_set = (
      set(AttributionVisualizationKeys.values()).difference(
          set(in_md.visualization.keys())))
  if required_keys_not_set:
    raise ValueError("Required key(s) not specified in 'visualization': %s." %
                     repr(required_keys_not_set))

  # Check that the TYPE key has a valid value.
  viz_type = in_md.visualization[AttributionVisualizationRequiredKeys.TYPE]
  validation_utils.validate_is_in(viz_type, "type",
                                  AttributionVisualizationTypes.values(),
                                  "IG input visualization config")
  for viz in AttributionVisualizationTypes.values():
    logging.debug(
        "XAI Validation :: Metadata: [IG input visualization config] "
        "All keys must be valid in visualization config. For `type` = `%s`, "
        "valid key set: `%s`", viz,
        repr(sorted(map(str, AttributionVisualizationKeys.values(viz)))))
  # Check that all keys set under visualization correspond to the expected
  # set of keys for the type of visualization being configured.
  invalid_keys_provided = set(in_md.visualization.keys()).difference(
      set(AttributionVisualizationKeys.values(viz_type)))
  if invalid_keys_provided:
    raise ValueError(
        "Invalid key(s) '%s' provided for visualization '%s'. Must be: '%s'."
        % (repr(invalid_keys_provided), viz_type,
           repr(AttributionVisualizationKeys.values(viz_type))))


def _validate_visualization_clip_range(in_md: InputMetadata):
  """Checks that the visualization clip range is valid."""
  logging.debug("XAI Validation :: Metadata: [IG input visualization config] "
                "Both `clip_below_percentile` and "
                "`clip_above_percentile` should be in range [0,100], and "
                "`clip_below_percentile` < `clip_above_percentile`.")
  if PixelsVisualizationKeys.CLIP_BELOW_PERCENTILE in in_md.visualization:
    clip_below_percentile = in_md.visualization[
        PixelsVisualizationKeys.CLIP_BELOW_PERCENTILE]
    if clip_below_percentile > 100 or clip_below_percentile < 0:
      raise ValueError(
          "Invalid value for clip_below_percentile: "
          "%s clip_below_percentile must be in range [0,100]."
          % repr(clip_below_percentile))
  if PixelsVisualizationKeys.CLIP_ABOVE_PERCENTILE in in_md.visualization:
    clip_above_percentile = in_md.visualization[
        PixelsVisualizationKeys.CLIP_ABOVE_PERCENTILE]
    if clip_above_percentile > 100 or clip_above_percentile < 0:
      raise ValueError(
          "Invalid value for clip_above_percentile: "
          "%s clip_above_percentile must be in range [0,100]."
          % repr(clip_above_percentile))
  #  clip_below_percentile should be smaller than clip_above_percentile
  if (PixelsVisualizationKeys.CLIP_ABOVE_PERCENTILE in in_md.visualization
      and PixelsVisualizationKeys.CLIP_BELOW_PERCENTILE
      in in_md.visualization):
    clip_above_percentile = in_md.visualization[
        PixelsVisualizationKeys.CLIP_ABOVE_PERCENTILE]
    clip_below_percentile = in_md.visualization[
        PixelsVisualizationKeys.CLIP_BELOW_PERCENTILE]
    if clip_below_percentile >= clip_above_percentile:
      raise ValueError(
          "Invalid values for clip_below_percentile, "
          "clip_above_percentile: %s clip_below_percentile must be smaller "
          "than clip_above_percentile."
          % repr((clip_above_percentile, clip_below_percentile)))


def validate_integrated_gradients(explain_md: ExplainMetadata):
  validate_visualization(explain_md)
  logging.debug("XAI Validation :: Metadata: [IG] "
                "`encoded_tensor_name` must be provided when "
                "input tensor is sparse for integrated gradient.")
  for in_md in explain_md.inputs:
    if (in_md.indices_tensor_name and in_md.dense_shape_tensor_name and
        not in_md.encoded_tensor_name):
      raise ValueError("encoded_tensor_name must be provided when "
                       "input tensor is sparse for integrated gradient.")


def validate_modality(explain_md: ExplainMetadata, expected_modality: str):
  for input_md in explain_md.inputs:
    if input_md.modality != expected_modality:
      raise ValueError("Input modality must be {}, got {} for input {}.".format(
          expected_modality, input_md.modality, input_md.name))
