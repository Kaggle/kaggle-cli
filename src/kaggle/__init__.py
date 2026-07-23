# coding=utf-8
from __future__ import absolute_import
from kaggle.api.kaggle_api_extended import KaggleApi

__version__ = "2.2.4"

api = KaggleApi()
try:
    api.authenticate()
except (Exception, SystemExit):
    pass
