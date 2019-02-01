from mlworkflow.data_freezing import (CallFreezer, ImageSaver, Pickleb64Freezer,
    PickleSaver, RelModulesFreezer)
from mlworkflow.file_handling import _format_filename
from mlworkflow.json_handling import djson_dump, djsonc_loads
from abc import ABCMeta, abstractmethod
from collections import ChainMap
from functools import wraps
import os


class _Provider(metaclass=ABCMeta):
    call = property(CallFreezer)
    image = png = property(ImageSaver)
    modules = property(RelModulesFreezer)
    pickle = property(PickleSaver)
    pickleb64 = property(Pickleb64Freezer)


class DataCollection(ChainMap, _Provider):
    """A class for recording experimental results
    """

    def add_metadata(self, dic):
        filename = self.filename if isinstance(self, DataCollection) else self
        assert isinstance(dic, dict), ("metadata must take the form of a "
                                       "dictionary")
        with open(filename+"_", "a") as file:
            djson_dump(dic, file, separators=(',',':'))
            file.write("\n")
            file.flush()

    @staticmethod
    def _read_json(file):
        while True:
            s = file.readline()
            if not s:
                break
            yield djsonc_loads(s)

    def get_metadata(self):
        filename = self.filename if isinstance(self, DataCollection) else self
        metadata = {}
        try:
            with open(filename+"_", "r") as file:
                for obj in DataCollection._read_json(file):
                    metadata.update(obj)
        except FileNotFoundError:
            pass
        return metadata

    @staticmethod
    def load_file(filename):
        with open(filename, "r") as file:
            return DataCollection._load_file_from_fp(file, filename)

    @staticmethod
    def _load_file_from_fp(file, filename):
        cum = {}
        data = []
        for obj in DataCollection._read_json(file):
            cum = {**cum, **obj}  # Cumulate fields
            data.append(_CheckPointWrapper(cum))  # Wrap
        return _CheckPointFileWrapper(data, filename=filename)

    def __init__(self, filename="{}.json", append=False):
        self._sparse = {}
        self._cumulated = {}
        super().__init__(self._sparse, self._cumulated)

        self.filename = _format_filename(filename)
        if os.path.exists(self.filename):
            assert append, ("{} already exists, append option is necessary to continue"
                            .format(self.filename))
            self.file = open(self.filename, "r+")
            self.history = DataCollection._load_file_from_fp(self.file, self.filename)
            self._cumulated.update(self.history[-1])
        else:
            self.file = open(self.filename, "w")
            self.history = _CheckPointFileWrapper([], filename=self.filename)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.history.__getitem__(key)
        if isinstance(key, list):
            sup = super()
            return [sup.__getitem__(k) for k in key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            assert len(key) == len(value)
            for k, v in zip(key, value):
                super().__setitem__(k, v)
        else:
            assert isinstance(key, str)
            super().__setitem__(key, value)

    @property
    def iteration(self):
        return len(self.history)

    @property
    def history_(self):
        return _CheckPointFileWrapper(self.history+[_CheckPointWrapper({**self._cumulated, **self._sparse})],
                                      filename=self.filename)

    def checkpoint(self):
        sparse = self._sparse
        cumulated = self._cumulated
        # Write sparse to file
        djson_dump(sparse, self.file, separators=(',',':'))
        self.file.write("\n")
        self.file.flush()
        # Update cumulated and history with a frozen version
        cumulated.update(sparse)
        frozen = cumulated.copy()
        self.history.append(_CheckPointWrapper(frozen))
        sparse.clear()

    @property
    def dirname(self):
        self.__dict__["dirname"] = os.path.dirname(self.filename)
        return self.dirname


class _CheckPointWrapper(dict):
    """"Add multiple and optional selections for a dict """
    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().get(key.start, key.stop)
        if isinstance(key, list):
            sup = super()
            return [sup.__getitem__(k) for k in key]
        return super().__getitem__(key)


class _CheckPointFileWrapper(list, _Provider):
    """Add slice selection for a list of CheckPointWrapper"""
    def __init__(self, *args, filename):
        super().__init__(*args)
        self.filename = filename

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2, ("Key tuple must be of length 2,"
                                   "got {!r}".format(key))
            key0 = key[0]
            key1 = key[1]
            if isinstance(key0, slice):
                sup = super()
                return [l[key1] for l in sup.__getitem__(key0)]
            return super().__getitem__(key0)[key1]
        return super().__getitem__(key)

    @property
    def dirname(self):
        dirname = os.path.dirname(self.filename)
        self.__dict__["dirname"] = dirname
        return dirname
