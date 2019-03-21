import connexion
import six

from swagger_server.models.add_predict_response import AddPredictResponse  # noqa: E501
from swagger_server.models.add_train_response import AddTrainResponse  # noqa: E501
from swagger_server.models.predict_status import PredictStatus  # noqa: E501
from swagger_server.models.train_status import TrainStatus  # noqa: E501
from swagger_server import util


def add_predict(predictData):  # noqa: E501
    """Predict by a new data

     # noqa: E501

    :param predictData: Predict data.
    :type predictData: werkzeug.datastructures.FileStorage

    :rtype: AddPredictResponse
    """
    return 'do some magic!'


def add_train(trainData):  # noqa: E501
    """Train by a new data

     # noqa: E501

    :param trainData: Train data.
    :type trainData: werkzeug.datastructures.FileStorage

    :rtype: AddTrainResponse
    """
    return 'do some magic!'


def get_predict_result(predictId):  # noqa: E501
    """predictの結果ファイルを取得する。

     # noqa: E501

    :param predictId: 特定のPredict IDを指定。
    :type predictId: str

    :rtype: file
    """
    return 'do some magic!'


def get_predict_status(predictId=None):  # noqa: E501
    """predictのステータスを取得する。

     # noqa: E501

    :param predictId: 特定のPredict IDを指定。省略した場合は全Predict IDのステータスを返却する。
    :type predictId: str

    :rtype: PredictStatus
    """
    return 'do some magic!'


def get_train_status(trainId=None):  # noqa: E501
    """trainのステータスを取得する。

     # noqa: E501

    :param trainId: 特定のTrain IDを指定。省略した場合は全Train IDのステータスを返却する。
    :type trainId: str

    :rtype: TrainStatus
    """
    return 'do some magic!'
