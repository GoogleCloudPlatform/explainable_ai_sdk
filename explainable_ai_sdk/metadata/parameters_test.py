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

"""Tests for google3.third_party.explainable_ai_sdk.sdk.metadata.parameters."""
import tensorflow as tf
from explainable_ai_sdk.metadata import parameters


class ParametersTest(tf.test.TestCase):

  def test_visualization_params_asdict_no_values_set(self):
    params = parameters.VisualizationParameters()
    self.assertEmpty(params.asdict())

  def test_visualization_params_asdict_some_values_set(self):
    params = parameters.VisualizationParameters(
        type=parameters.VisualizationType.OUTLINES, clip_above_percentile=0.25)
    d = params.asdict()
    self.assertIn("type", d)
    self.assertIn("outlines", d["type"])
    self.assertIn("clip_above_percentile", d)
    self.assertAlmostEqual(0.25, d["clip_above_percentile"])
    self.assertNotIn("overlay_type", d)

  def test_visualization_params_asdict_enums_to_text(self):
    params = parameters.VisualizationParameters(
        overlay_type=parameters.OverlayType.NONE,
        color_map=parameters.ColorMap.VIRIDIS)
    d = params.asdict()
    self.assertEqual("viridis", d["color_map"])
    self.assertEqual("none", d["overlay_type"])

  def test_domain_info_asdict(self):
    domain_info = parameters.DomainInfo(min=0.1, max=0.9, original_mean=6)
    d = domain_info.asdict()
    self.assertEqual(0.1, d["min"])
    self.assertEqual(6, d["original_mean"])
    self.assertNotIn("original_stddev", d)


if __name__ == "__main__":
  tf.test.main()
