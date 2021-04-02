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

"""Utilities for runtime variable validations."""

from typing import get_type_hints, List, Dict, Union, Any
from absl import logging


def validate_is_instance(var: Any,
                         var_name: str,
                         instance_type: Any,
                         class_name: str = None,
                         log_metadata_validation_failures: bool = True) -> None:
  """Validates the type of the given variable and raises exception if not pass.

  Args:
    var: The variable to verify.
    var_name: The name of the variable.
    instance_type: The expected variable type.
    class_name: The class that the variable belongs to.
    log_metadata_validation_failures: Metadata validation failures will be sent
    to logging.debug() if set to True.
  Raises:
    ValueError: If the given variable is not of expected type.
  """
  if var is None:
    return
  splits = str(instance_type).split("<class ")[-1].split("'")
  if len(splits) > 1:
    print_type = splits[1]
  else:
    print_type = splits[0]
  if log_metadata_validation_failures:
    if class_name is None:
      logging.debug(
          "XAI Validation :: Metadata: Variable `%s` should be of type `%s`",
          var_name, print_type)
    else:
      logging.debug(
          "XAI Validation :: Metadata: [%s] Variable `%s` should be of type "
          "`%s`", class_name, var_name, print_type)
  if not isinstance(var, instance_type):
    raise TypeError("{} must be of type {}. Got {}".format(
        var_name, str(instance_type), str(type(var))))


def validate_object_init_type_hint(obj: object,
                                   log_metadata_validation_failures: bool = True
                                  ) -> None:
  """Checks if the variables of the given object matched the type hints of init.

  Args:
    obj: The object to verify.
    log_metadata_validation_failures: Metadata validation failures will be sent
    to logging.debug() if set to True.
  Raises:
    ValueError: If any of the given variable does not match type hint of init.
  """
  if not obj or not type(obj).__init__:
    return
  type_hint_dict = get_type_hints(type(obj).__init__)
  if not isinstance(type_hint_dict, dict):
    return
  for var_name, instance_type in type_hint_dict.items():
    generic_alias_map = {List: list, Dict: dict, list: list, dict: dict}
    if hasattr(instance_type, "__origin__"):
      if instance_type.__origin__ is Union:
        # optional argument, [1] is None
        instance_type = instance_type.__args__[0]
        if hasattr(instance_type, "__origin__"):
          instance_type = generic_alias_map[instance_type.__origin__]
      else:
        instance_type = generic_alias_map[instance_type.__origin__]
    if hasattr(obj, var_name):
      var = getattr(obj, var_name)
      validate_is_instance(var, var_name, instance_type,
                           type(obj).__name__, log_metadata_validation_failures)


def validate_is_in(var: Any,
                   var_name: str,
                   list_type: Any,
                   class_name: str = None,
                   log_metadata_validation_failures: bool = True) -> None:
  """Validates if the given variable is a member of the given list-type object.

  Args:
    var: The variable to verify.
    var_name: The name of the variable.
    list_type: The given list-type object.
    class_name: The class that the variable belongs to.
    log_metadata_validation_failures: Metadata validation failures will be sent
    to logging.debug() if set to True.

  Raises:
    ValueError: If the given variable is not a member of the list-type object.
  """
  if var is None:
    return
  sorted_list_type = sorted(map(str, list_type))
  if log_metadata_validation_failures:
    if class_name is None:
      logging.debug(
          "XAI Validation :: Metadata: Variable `%s` should be a member of "
          "`%s`", var_name, sorted_list_type)
    else:
      logging.debug(
          "XAI Validation :: Metadata: [%s] Variable `%s` should be a member "
          "of `%s`", class_name, var_name, sorted_list_type)
  if var not in list_type:
    raise ValueError("{} not in {}. Got {}.".format(
        var_name, sorted_list_type, var))
