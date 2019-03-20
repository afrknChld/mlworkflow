from mlworkflow.json_handling import Call

from abc import ABCMeta, abstractproperty, abstractmethod
import numpy as np
import os


class DataFreezer(metaclass=ABCMeta):
    def __init__(self, provider):
        self.provider = provider

    def _add_metadata(self, descriptor):
        return {"_frozen_by": self.name, **descriptor}

    def __setitem__(self, key, value):
        if isinstance(key, list):
            self.provider[key] = [self._add_metadata(self.freeze(k, v))
                                  for k, v in zip(key, value)]
        else:
            self.provider[key] = self._add_metadata(self.freeze(key, value))

    def __getitem__(self, key):
        descriptor = self.provider[key]
        return self._unfreeze(descriptor)

    def _unfreeze(self, descriptor):
        if isinstance(descriptor, list):
            return [self.unfreeze(d) for d in descriptor]
        return self.unfreeze(descriptor)

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def freeze(self, key, value):
        """Takes a key and a value to associate to it, returns a descriptor."""
        pass

    @abstractmethod
    def unfreeze(self, descriptor):
        """Takes a descriptor, returns the value.
        
        Should not rely on the provider as it may not be a data collection"""
        pass


class NoTypeFreezer(DataFreezer):
    def _add_metadata(self, descriptor):
        return descriptor


class CallFreezer(NoTypeFreezer):
    name = "call"
    def freeze(self, key, value):
        assert value["_call"] == "Call"
        return value
    
    def unfreeze(self, descriptor):
        return Call.instantiate(descriptor)


import pickle
import base64
class Pickleb64Freezer(DataFreezer):
    name = "pickleb64"
    def freeze(self, key, value):
        return {"value": base64.b64encode(pickle.dumps(value)).decode("utf-8")}
    
    def unfreeze(self, descriptor):
        return pickle.loads(base64.b64decode(descriptor["value"].encode("utf-8")))


class RelModulesFreezer(DataFreezer):
    name = "modules"
    def freeze(self, key, value):
        if isinstance(value, (set, list, tuple)):
            value = {v: v for v in value}
        assert isinstance(value, dict)
        return {"modules": value}

    def unfreeze(self, descriptor):
        path = os.path.relpath(self.provider.dirname)
        base = path.replace(os.sep, ".")
        redirs = descriptor["modules"]
        # <LEGACY>
        if isinstance(redirs, (list, tuple)):
            redirs = {r: r for r in redirs}
        # </LEGACY>
        redirs = {k: "{}.{}".format(base, v) for k, v in redirs.items()}
        return redirs


class DataSaver(DataFreezer):
    def freeze(self, key, value):
        assert key, "key cannot be empty"
        filename = "{}_{}_{}".format(self.provider.filename, self.provider.iteration, key)
        self.save(filename, value)
        return {"filename": os.path.basename(filename)}

    def unfreeze(self, descriptor):
        return self.load(os.path.join(self.provider.dirname, descriptor["filename"]))

    @abstractmethod
    def save(self, filename, value):
        pass

    @abstractmethod
    def load(self, filename):
        pass


class PickleSaver(DataSaver):
    name = "pickle"
    def save(self, filename, obj):
        with open(filename, "wb") as file:
            pickle.dump(obj, file)

    def load(self, filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


try:
    #pylint: disable=no-member
    import cv2
    class ImageSaver(DataSaver):
        name = "png"
        def save(self, filename, obj):
            cv2.imwrite(filename+".png", obj[...,::-1])

        def load(self, filename):
            return cv2.imread(filename+".png")[...,::-1]
except ImportError:
    ImageSaver = None
