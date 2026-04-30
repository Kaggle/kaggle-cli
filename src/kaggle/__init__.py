# coding=utf-8
from __future__ import absolute_import
from kaggle.api.kaggle_api_extended import KaggleApi

__version__ = "2.1.0"

api = KaggleApi()
api.authenticate()
