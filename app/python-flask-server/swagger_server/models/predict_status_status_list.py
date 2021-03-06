# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class PredictStatusStatusList(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, predict_id: str=None, status: str=None, message: str=None):  # noqa: E501
        """PredictStatusStatusList - a model defined in Swagger

        :param predict_id: The predict_id of this PredictStatusStatusList.  # noqa: E501
        :type predict_id: str
        :param status: The status of this PredictStatusStatusList.  # noqa: E501
        :type status: str
        :param message: The message of this PredictStatusStatusList.  # noqa: E501
        :type message: str
        """
        self.swagger_types = {
            'predict_id': str,
            'status': str,
            'message': str
        }

        self.attribute_map = {
            'predict_id': 'predictId',
            'status': 'status',
            'message': 'message'
        }

        self._predict_id = predict_id
        self._status = status
        self._message = message

    @classmethod
    def from_dict(cls, dikt) -> 'PredictStatusStatusList':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The PredictStatus_status_list of this PredictStatusStatusList.  # noqa: E501
        :rtype: PredictStatusStatusList
        """
        return util.deserialize_model(dikt, cls)

    @property
    def predict_id(self) -> str:
        """Gets the predict_id of this PredictStatusStatusList.


        :return: The predict_id of this PredictStatusStatusList.
        :rtype: str
        """
        return self._predict_id

    @predict_id.setter
    def predict_id(self, predict_id: str):
        """Sets the predict_id of this PredictStatusStatusList.


        :param predict_id: The predict_id of this PredictStatusStatusList.
        :type predict_id: str
        """
        if predict_id is None:
            raise ValueError("Invalid value for `predict_id`, must not be `None`")  # noqa: E501

        self._predict_id = predict_id

    @property
    def status(self) -> str:
        """Gets the status of this PredictStatusStatusList.


        :return: The status of this PredictStatusStatusList.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this PredictStatusStatusList.


        :param status: The status of this PredictStatusStatusList.
        :type status: str
        """
        if status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")  # noqa: E501

        self._status = status

    @property
    def message(self) -> str:
        """Gets the message of this PredictStatusStatusList.


        :return: The message of this PredictStatusStatusList.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this PredictStatusStatusList.


        :param message: The message of this PredictStatusStatusList.
        :type message: str
        """

        self._message = message
