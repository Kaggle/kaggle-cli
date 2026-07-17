#!/usr/bin/python
#
# Copyright 2024 Kaggle Inc
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

# coding: utf-8

import pprint
import re  # noqa: F401

import six
from typing import Any
from kagglesdk.datasets.types.dataset_api_service import ApiDatasetColumn


class UploadFile(object):
    """
    Attributes:
      column_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """

    column_types = {"token": "str", "description": "str", "columns": "list[ApiDatasetColumn]"}

    attribute_map = {"token": "token", "description": "description", "columns": "columns"}

    def __init__(self, token=None, description=None, columns=None):  # noqa: E501
        """UploadFile - a model defined in Swagger"""  # noqa: E501

        self._token = None
        self._description = None
        self._columns = None
        self.discriminator = None

        if token is not None:
            self.token = token
        if description is not None:
            self.description = description
        if columns is not None:
            self.columns = columns

    @property
    def token(self):
        """Gets the token of this UploadFile.  # noqa: E501.

        A token referencing a specific file upload that can be used across
        requests  # noqa: E501

        :return: The token of this UploadFile.  # noqa: E501
        :rtype: str
        """
        return self._token

    @token.setter
    def token(self, token):
        """Sets the token of this UploadFile.

        A token referencing a specific file upload that can be used across
        requests  # noqa: E501

        :param token: The token of this UploadFile.  # noqa: E501
        :type: str
        """

        self._token = token

    @property
    def description(self):
        """Gets the description of this UploadFile.  # noqa: E501.

        The file description  # noqa: E501

        :return: The description of this UploadFile.  # noqa: E501
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """Sets the description of this UploadFile.

        The file description  # noqa: E501

        :param description: The description of this UploadFile. # noqa: E501
        :type: str
        """

        self._description = description

    @property
    def columns(self):
        """Gets the columns of this UploadFile.  # noqa: E501.

        A list of dataset column metadata  # noqa: E501

        :return: The columns of this UploadFile.  # noqa: E501
        :rtype: list[ApiDatasetColumn]
        """
        return self._columns

    @columns.setter
    def columns(self, columns):
        """Sets the columns of this UploadFile.

        A list of dataset column metadata  # noqa: E501

        :param columns: The columns of this UploadFile.  # noqa: E501
        :type: list[ApiDatasetColumn]
        """

        self._columns = columns

    def to_dict(self):
        """Returns the model properties as a dict."""
        result: dict[str, Any] = {}

        for attr, _ in six.iteritems(self.column_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict()) if hasattr(item[1], "to_dict") else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal."""
        if not isinstance(other, UploadFile):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal."""
        return not self == other
