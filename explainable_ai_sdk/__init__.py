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

"""Init module for Explainability SDK.

Programs that want to use metadata builders and model factory without having to
import them individually can import this file:

import explainable_ai_sdk
"""

from explainable_ai_sdk.model.model_factory import load_model_from_ai_platform
from explainable_ai_sdk.model.model_factory import load_model_from_local_path
from explainable_ai_sdk.model.model_factory import load_model_from_unified_ai_platform
