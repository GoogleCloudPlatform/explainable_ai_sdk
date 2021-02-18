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


"""Model classes for obtaining explanations."""

import abc

from typing import Any, Dict, List

from explainable_ai_sdk.model import configs
from explainable_ai_sdk.model import explanation

_ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class Model(_ABC):
  """Base class for model related operations in SDK."""

  @abc.abstractmethod
  def predict(self, instances: List[Any]) -> List[Dict[Any, Any]]:
    """Calls prediction services/libraries with the given instances.

    Args:
       instances: A list of instances for getting predictions.

    Returns:
       A list of dictionaries to represent predictions.
    """
    pass

  @abc.abstractmethod
  def explain(
      self,
      instances: List[Any],
      params: configs.AttributionParameters = None
  ) -> List[explanation.Explanation]:
    """Calls explanation services/libraries with the given instances.

    Args:
       instances: A list of instances for getting explanations.
       params: Overridable parameters for the explain call. If not provided,
         parameters default to what was already set before.

    Returns:
       A dictionary to map input to its corresponding list of Explanation
         objects.
    """
    pass
