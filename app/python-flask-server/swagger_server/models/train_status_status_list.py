# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class TrainStatusStatusList(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, train_id: str=None, status: str=None, message: str=None):  # noqa: E501
        """TrainStatusStatusList - a model defined in Swagger

        :param train_id: The train_id of this TrainStatusStatusList.  # noqa: E501
        :type train_id: str
        :param status: The status of this TrainStatusStatusList.  # noqa: E501
        :type status: str
        :param message: The message of this TrainStatusStatusList.  # noqa: E501
        :type message: str
        """
        self.swagger_types = {
            'train_id': str,
            'status': str,
            'message': str
        }

        self.attribute_map = {
            'train_id': 'trainId',
            'status': 'status',
            'message': 'message'
        }

        self._train_id = train_id
        self._status = status
        self._message = message

    @classmethod
    def from_dict(cls, dikt) -> 'TrainStatusStatusList':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The TrainStatus_status_list of this TrainStatusStatusList.  # noqa: E501
        :rtype: TrainStatusStatusList
        """
        return util.deserialize_model(dikt, cls)

    @property
    def train_id(self) -> str:
        """Gets the train_id of this TrainStatusStatusList.


        :return: The train_id of this TrainStatusStatusList.
        :rtype: str
        """
        return self._train_id

    @train_id.setter
    def train_id(self, train_id: str):
        """Sets the train_id of this TrainStatusStatusList.


        :param train_id: The train_id of this TrainStatusStatusList.
        :type train_id: str
        """
        if train_id is None:
            raise ValueError("Invalid value for `train_id`, must not be `None`")  # noqa: E501

        self._train_id = train_id

    @property
    def status(self) -> str:
        """Gets the status of this TrainStatusStatusList.


        :return: The status of this TrainStatusStatusList.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this TrainStatusStatusList.


        :param status: The status of this TrainStatusStatusList.
        :type status: str
        """
        if status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")  # noqa: E501

        self._status = status

    @property
    def message(self) -> str:
        """Gets the message of this TrainStatusStatusList.


        :return: The message of this TrainStatusStatusList.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this TrainStatusStatusList.


        :param message: The message of this TrainStatusStatusList.
        :type message: str
        """

        self._message = message