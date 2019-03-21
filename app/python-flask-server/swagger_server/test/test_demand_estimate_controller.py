# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.add_predict_response import AddPredictResponse  # noqa: E501
from swagger_server.models.add_train_response import AddTrainResponse  # noqa: E501
from swagger_server.models.predict_status import PredictStatus  # noqa: E501
from swagger_server.models.train_status import TrainStatus  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDemandEstimateController(BaseTestCase):
    """DemandEstimateController integration test stubs"""

    def test_add_predict(self):
        """Test case for add_predict

        Predict by a new data
        """
        data = dict(predictData=(BytesIO(b'some file data'), 'file.txt'))
        response = self.client.open(
            '/v0.1/predict',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_add_train(self):
        """Test case for add_train

        Train by a new data
        """
        data = dict(trainData=(BytesIO(b'some file data'), 'file.txt'))
        response = self.client.open(
            '/v0.1/train',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_predict_result(self):
        """Test case for get_predict_result

        predictの結果ファイルを取得する。
        """
        query_string = [('predictId', 'predictId_example')]
        response = self.client.open(
            '/v0.1/predict/result',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_predict_status(self):
        """Test case for get_predict_status

        predictのステータスを取得する。
        """
        query_string = [('predictId', 'predictId_example')]
        response = self.client.open(
            '/v0.1/predict',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_train_status(self):
        """Test case for get_train_status

        trainのステータスを取得する。
        """
        query_string = [('trainId', 'trainId_example')]
        response = self.client.open(
            '/v0.1/train',
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
