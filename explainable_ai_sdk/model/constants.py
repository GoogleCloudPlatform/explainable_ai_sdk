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


"""Constants for SDK models."""
from explainable_ai_sdk import version

# Modality related constants
ALL_MODALITY = 'all'
TABULAR_MODALITY = 'tabular'

# HTTP related constants
DEFAULT_TIMEOUT = 1200
USER_AGENT_FOR_CAIP_TRACKING = 'xai-sdk/' + version.__version__

CAIP_API_ENDPOINT = 'https://ml.googleapis.com/'
CAIP_API_REGION_ENDPOINT = 'https://{region}-ml.googleapis.com/'
CAIP_API_ENDPOINT_VERSION = 'v1'

UCAIP_PREDICTION_API_ENDPOINT = 'https://{region}-prediction-aiplatform.googleapis.com'
UCAIP_API_ENDPOINT_VERSION = 'v1beta1'

AIP_ENDPOINT_OVERRIDE = 'CLOUDSDK_API_ENDPOINT_OVERRIDES_ML'
