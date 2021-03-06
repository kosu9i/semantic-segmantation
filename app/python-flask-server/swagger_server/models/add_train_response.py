# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class AddTrainResponse(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, code: int=None, message: str=None, train_id: str=None):  # noqa: E501
        """AddTrainResponse - a model defined in Swagger

        :param code: The code of this AddTrainResponse.  # noqa: E501
        :type code: int
        :param message: The message of this AddTrainResponse.  # noqa: E501
        :type message: str
        :param train_id: The train_id of this AddTrainResponse.  # noqa: E501
        :type train_id: str
        """
        self.swagger_types = {
            'code': int,
            'message': str,
            'train_id': str
        }

        self.attribute_map = {
            'code': 'code',
            'message': 'message',
            'train_id': 'trainId'
        }

        self._code = code
        self._message = message
        self._train_id = train_id

    @classmethod
    def from_dict(cls, dikt) -> 'AddTrainResponse':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The AddTrainResponse of this AddTrainResponse.  # noqa: E501
        :rtype: AddTrainResponse
        """
        return util.deserialize_model(dikt, cls)

    @property
    def code(self) -> int:
        """Gets the code of this AddTrainResponse.


        :return: The code of this AddTrainResponse.
        :rtype: int
        """
        return self._code

    @code.setter
    def code(self, code: int):
        """Sets the code of this AddTrainResponse.


        :param code: The code of this AddTrainResponse.
        :type code: int
        """
        if code is None:
            raise ValueError("Invalid value for `code`, must not be `None`")  # noqa: E501

        self._code = code

    @property
    def message(self) -> str:
        """Gets the message of this AddTrainResponse.


        :return: The message of this AddTrainResponse.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this AddTrainResponse.


        :param message: The message of this AddTrainResponse.
        :type message: str
        """
        if message is None:
            raise ValueError("Invalid value for `message`, must not be `None`")  # noqa: E501

        self._message = message

    @property
    def train_id(self) -> str:
        """Gets the train_id of this AddTrainResponse.


        :return: The train_id of this AddTrainResponse.
        :rtype: str
        """
        return self._train_id

    @train_id.setter
    def train_id(self, train_id: str):
        """Sets the train_id of this AddTrainResponse.


        :param train_id: The train_id of this AddTrainResponse.
        :type train_id: str
        """

        self._train_id = train_id
