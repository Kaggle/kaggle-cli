# coding=utf-8
from __future__ import absolute_import
import os
from kaggle.api.kaggle_api_extended import KaggleApi

__version__ = "1.8.4"

enable_oauth = os.environ.get("KAGGLE_ENABLE_OAUTH") in ("1", "true", "yes")
api = KaggleApi(enable_oauth=enable_oauth)
api.authenticate()
