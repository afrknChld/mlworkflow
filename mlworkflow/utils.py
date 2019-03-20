"""Legacy module"""

from mlworkflow.miscellaneous import DictObject


def put_attrs(f, dic):
    for k, v in dic.items():
        setattr(f, k, v)
