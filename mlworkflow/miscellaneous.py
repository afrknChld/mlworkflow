from mlworkflow.file_handling import _format_filename
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
from collections import deque
from functools import wraps
import pickle
import os

_no_value = object()
gen_id = _format_filename


class DictObject(dict):
    def __new__(cls, *args, **kwargs):
        dict_object = super().__new__(cls, *args, **kwargs)
        dict_object.__dict__ = dict_object
        return dict_object

    def __repr__(self):
        return "{}({})".format(self.__class__.__qualname__,
                               super().__repr__())

    @classmethod
    def from_dict(cls, dic):
        """__init__ may not simply copy the argument in the object. In order
        to directly feed the dictionary, from_dict can be used
        """
        dict_object = cls.__new__(cls)
        dict_object.update(dic)
        return dict_object

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        for k, v in self.items():
            copy[k] = v
        return copy
    copy = __copy__

    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        copy = self.__class__.__new__(self.__class__)
        for k, v in self.items():
            copy[k] = deepcopy(v, memo)
        return copy
    deepcopy = __deepcopy__

    def __reduce__(self):
        return DictObject._v0, (self.__class__.__module__,
                                self.__class__.__qualname__,
                                dict(self))

    @staticmethod
    def _v0(module, qualname, items):
        try:
            from importlib import import_module
            klass = import_module(module)
            names = qualname.split(".")
            for name in names:
                klass = getattr(klass, name)
        except (ImportError, AttributeError):
            klass = DictObject
        target = klass.from_dict(items)
        return target


def pickle_cache(name):
    def _decorator(f):
        @wraps(f)
        def wrapper(**kwargs):
            filename = name.format(**kwargs)
            if os.path.exists(filename):
                with open(filename, "rb") as file:
                    return pickle.load(file)
            result = f(**kwargs)
            with open(filename, "wb") as file:
                pickle.dump(result, file)
            return result
        return wrapper
    return _decorator


class SideRunner:
    def __init__(self):
        self.pool = ThreadPool(1)
        self.pending = deque()

    def run_async(self, f):
        handle = self.pool.apply_async(f)
        self.pending.append(handle)
        return handle

    def wait_for_complete(self, i):
        j = i+1 if i != -1 else None
        for p in self.pending[i:j]:
            p.wait()

    def collect_runs(self):
        lst = [handle.get() for handle in self.pending]
        self.pending.clear()
        return lst

    def yield_async(self, gen, in_advance=1):
        pending = deque()
        def consume(gen):
            return next(gen, _no_value)
        for _ in range(in_advance):
            pending.append(self.pool.apply_async(consume, args=(gen,)))
        while True:
            pending.append(self.pool.apply_async(consume, args=(gen,)))
            p = pending.popleft().get()
            if p is _no_value:
                break
            yield p

    def __del__(self):
        self.pool.close()
        self.pool.join()


@contextmanager
def seed(random, seed):
    if hasattr(random, "get_state"):
        old_state = random.get_state()
        random.seed(seed)
        yield random
        random.set_state(old_state)
    elif hasattr(random, "getstate"):
        old_state = random.getstate()
        random.seed(seed)
        yield random
        random.setstate(old_state)
    else:
        raise Exception("Random object not recognized")