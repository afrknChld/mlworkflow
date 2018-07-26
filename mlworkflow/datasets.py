from pickle import Pickler, _Unpickler as Unpickler
from mlworkflow.utils import warning
from abc import ABCMeta, abstractmethod
import numpy as np
import functools
import sys
import os

import weakref

# For ParallelizedDataset
from multiprocessing import Process, Queue, Pipe
from collections import deque, ChainMap
from multiprocessing.pool import Pool
from contextlib import contextmanager
import math


def chunkify(iterable, n):
    """Return a generator providing chunks (lists of size n) of the iterable.

    >>> tuple(chunkify(range(10), 5))  # len(iterable) % n == 0
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    >>> tuple(chunkify(range(12), 5))  # len(iterable) % n != 0
    ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11])
    >>> tuple(chunkify([], 100))       # Empty iterable example
    ([],)
    """
    offset = 0
    ret = [None]*n  # filled the majority of the time => avoid growing list
    i = -1
    for i, e in enumerate(iterable):
        if i - offset == n:  # yield complete sublist and create a new list
            yield ret
            offset += n
            ret = [None]*n
        ret[i - offset] = e
    yield ret[:i-offset+1]  # yield the incomplete subset ([] if i = -1)


class Dataset(metaclass=ABCMeta):
    """The base class for any dataset, provides the and batches methods from
    list_keys() and query_item(key)

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d.query([0, 1])
    (array(['Denzel', 'Tom'], ...), array(['Washington', 'Hanks'], ...))
    >>> list(d.batches([0, 1], 1))
    [(array(['Denzel'], ...), array(['Washington'], ...),
     (array(['Tom'], ...), array(['Hanks'], ...))]

    We can see the latter provides two batches
    """

    @abstractmethod
    def list_keys(self):
        pass

    @abstractmethod
    def query_item(self, key):
        """Returns a tuple for one item, typically (Xi, Yi), or (Xi,)
        """
        pass

    def query(self, keys, wrap=False):
        """Computes a batch, typically (X, Y) from the items (Xi, Yi) yielded
        by query_item(keys[i]).

        At this point, we consider keys is a list.
        """
        iterator = enumerate(keys)
        _, key0 = next(iterator)
        first = self.query_item(key0)
        if wrap:
            first = (first,)
        width = range(len(first))

        XYs = [[None]*len(keys) for j in width]
        for j in width:
            XYs[j][0] = first[j]

        for i, key in iterator:
            item = self.query_item(key)
            if wrap:
                item = (item,)
            for j in width:
                XYs[j][i] = item[j]
        return tuple(np.array(Xs) for Xs in XYs)

    def batches(self, keys, batch_size, **kwargs):
        """Compute batches to make one epoch of the given keys

        Remember to perform the shuffling of the keys before!
        """
        for key_chunk in chunkify(keys, batch_size):
            yield self.query(key_chunk, **kwargs)

    def balanced_batches(self, split_keys, batch_size, **kwargs):
        """Compute balanced batches to make one epoch with respect to the
        smallest list of keys.

        Remember to perform the shuffling of the keys before!
        """
        sub_sizes = batch_size // len(split_keys)
        assert sub_sizes * len(split_keys) == batch_size
        # one generator of chunks for each
        batched_keys = [chunkify(keys, sub_sizes)
                        for keys in split_keys]
        # one parallel a tuple with 1 chunk of each
        for parallel in zip(*batched_keys):
            min_length = min(len(p) for p in parallel)
            if min_length != batch_size:
                parallel = [p[:min_length] for p in parallel]
            Xs, Ys = self.query(sum(parallel, []), **kwargs)
            yield Xs, Ys

    @property
    def parent_dataset(self):
        return self._parent_dataset

    @parent_dataset.setter
    def parent_dataset(self, parent_dataset):
        self._parent_dataset = parent_dataset
        if self._parent_dataset is not None:
            self._context = parent_dataset.context.new_child(self.context.maps[0])

    dataset = parent_dataset  # TODO: Remove this alias

    @property
    def context(self):
        try:
            return self._context
        except AttributeError:
            self._context = ChainMap()
            return self._context

    def items_equality(self, a, b):
        return self._recursive_equality(a, b)

    def _recursive_equality(self, a, b):
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                return False
            for a_, b_ in zip(a, b):
                if not self._recursive_equality(a_, b_):
                    return False
            return True
        if isinstance(a, dict) and isinstance(b, dict):
            keys = a.keys()
            if keys != b.keys():
                return False
            for key in keys:
                if not self._recursive_equality(a[key], b[key]):
                    return False
            return True
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        return a == b


def replace_method(obj, name=None):
    """A decorator for overriding the method of an object by a function

    >>> class X:
    ...     pass
    >>> x = X()
    >>> @replace_method(x)
    ... def test():
    ...    return "all right"
    >>> x.test()
    'all right'
    """
    def decorator(f):
        nonlocal name
        if name is None:
            name = f.__name__
        setattr(obj, name, f)
        return f
    return decorator


class _ListKeyWrapper(Dataset):
    def __init__(self, dataset, wrapper):
        self.dataset = dataset
        self.wrapper = wrapper

    def list_keys(self):
        yield from self.wrapper(self.dataset.list_keys())

    def query_item(self, key):
        return self.dataset.query_item(key)


@replace_method(Dataset, "tqdm")
def _tqdm(self, *args, lazy_keys=False, **kwargs):
    from tqdm import tqdm
    pre_tqdm = (lambda x: x) if lazy_keys else list
    return _ListKeyWrapper(self, lambda keys: tqdm(pre_tqdm(keys),
                                                   *args, **kwargs
                                                   )
                           )


@replace_method(Dataset, "tqdm_notebook")
def _tqdm_notebook(self, *args, lazy_keys=False, **kwargs):
    from tqdm import tqdm_notebook
    pre_tqdm = (lambda x: x) if lazy_keys else list
    return _ListKeyWrapper(self, lambda keys: tqdm_notebook(pre_tqdm(keys),
                                                            *args, **kwargs
                                                            )
                           )


class TransformedDataset(Dataset):
    """A dataset that passes yielded items through transforms

    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = TransformedDataset(d, [lambda x: (x[0][:1]+".", x[1])])
    >>> d.query([0, 1])
    (array(['D.', 'T.'], ...), array(['Washington', 'Hanks'], ...))
    """
    def __init__(self, dataset, transforms=[]):
        """Creates a dataset performing operations for modifying another"""
        self.dataset = dataset
        self.transforms = [(t, getattr(t, "needs_key", False))
                           for t in transforms]

    def list_keys(self):
        return self.dataset.list_keys()

    def query_item(self, key):
        item = self.dataset.query_item(key)
        for transform, needs_key in self.transforms:
            if needs_key:
                item = transform(key, item)
            else:
                item = transform(item)
        return item

    def add_transform(self, transform=None, *, needs_key=False):
        _needs_key = needs_key
        def add_transform(transform):
            needs_key = _needs_key
            if not needs_key:
                needs_key = getattr(transform, "needs_key", False)
            item = (transform, needs_key)
            self.transforms.append(item)
            return transform
        if transform is not None:
            return add_transform(transform)
        return add_transform

    def add_transforms(self, transforms):
        self.transforms.extend((t, getattr(t, "needs_key", False))
                               for t in transforms)


class CacheLastDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = (None, None)

    def list_keys(self):
        return self.dataset.list_keys()

    def after_cache_miss(self, key, item):
        pass

    def query_item(self, key):
        cached_key, item = self.cache
        if key != cached_key:
            item = self.dataset.query_item(key)
            self.cache = (key, item)
            self.after_cache_miss(key, item)
        return item


class AugmentedDataset(Dataset, metaclass=ABCMeta):
    """ "Augments" a dataset in the sense that it can produce many items from
    one item of the dataset.

    >>> class PermutingDataset(AugmentedDataset):
    ...     def augment(self, root_key, root_item):
    ...         yield (root_key, 0), root_item
    ...         yield (root_key, 1), root_item[::-1]
    >>> d = DictDataset({0: ("Denzel", "Washington"), 1: ("Tom", "Hanks")})
    >>> d = PermutingDataset(d)
    >>> new_keys = list(d.list_keys())
    >>> new_keys
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    >>> d.query(new_keys)
    (array(['Denzel', 'Washington', 'Tom', 'Hanks'], ...),
     array(['Washington', 'Denzel', 'Hanks', 'Tom'], ...))
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = (None, None)

    def optimize_query_order(self, dataset):
        old_query = dataset.query

        def query(keys):
            keys.sort()
            return old_query(keys)
        dataset.query = query

    def _augment(self, root_key):
        if self.cache[0] != root_key:
            root_item = self.dataset.query_item(root_key)
            new_items = dict(self.augment(root_key, root_item))
            self.cache = (root_key, new_items)
        return self.cache[1]

    def list_keys(self):
        for root_key in self.dataset.list_keys():
            yield from self._augment(root_key).keys()

    def query_item(self, key):
        return self._augment(key[0])[key]

    @abstractmethod
    def augment(self, root_key, root_item):
        pass


class CachedDataset(Dataset):
    """Creates a dataset caching the result of another"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def list_keys(self):
        return self.dataset.list_keys()

    def query_item(self, key):
        tup = self.cache.get(key, None)
        if tup is not None:
            return tup
        tup = self.dataset.query_item(key)
        self.cache[key] = tup
        return tup

    def _cached_keys(self):
        return self.cache.keys()

    def fill_forget(self):
        for key in self.dataset.list_keys():
            self.query_item(key)
        self.list_keys = self._cached_keys
        self.dataset = None
        return self


class PickledDataset(Dataset):
    """A dataset compacted on the disk with Pickle. For initial creation from
    an old dataset::

        in_mem_dataset = DictDataset({"a": 1, "b": 2, "c": 3})
        with open("file_path", "wb") as f:
            PickledDataset.create(in_mem_dataset, f)

    For using a PickledDataset::

        with open("file_path", "rb") as f:
            pd = PickledDataset(f)
            pd = TransformedDataset(pd, [lambda x, draw: (x, x)])
            X, Y = pd.query(pd.list_keys())
            model.fit(X, Y)
    """
    @staticmethod
    def create(dataset, file_handler, keys=None):
        if isinstance(file_handler, str):
            with open(file_handler, "wb") as file_handler:
                return PickledDataset.create(dataset, file_handler, keys=keys)
        index = {}
        pickler = Pickler(file_handler)
        # allocate space for index offset
        file_handler.seek(0)
        pickler.dump(1 << 65)  # 64 bits placeholder
        if keys is None:
            keys = dataset.list_keys()
        for key in keys:
            # pickle objects and build index
            index[key] = file_handler.tell()
            obj = dataset.query_item(key)
            pickler.dump(obj)
            pickler.memo.clear()
        # put index and record offset
        index_location = file_handler.tell()
        pickler.dump(index)
        # put context
        context = getattr(dataset, "_context", None)
        if context:
            pickler.dump({**context})
        # put index offset at the beginning of the file
        file_handler.seek(0)
        index_location ^= 1 << 65
        pickler.dump(index_location)

    def __init__(self, file_handler, offset_keys=False):
        if isinstance(file_handler, str):
            file_handler = open(file_handler, "rb")
        self.file_handler = file_handler
        self.offset_keys = offset_keys
        self.unpickler = unpickler = Unpickler(file_handler)

        # load the index offset then the index
        file_handler.seek(0)
        index_location = unpickler.load()
        index_location ^= 1 << 65
        file_handler.seek(index_location)
        self.index = unpickler.load()
        # try to load the context if any
        try:
            self._context = ChainMap(unpickler.load())
        except EOFError:
            pass

        if offset_keys:
            def list_keys():
                return self.index.values()
            self.list_keys = list_keys()

            def query_item(key):
                self.file_handler.seek(key)
                return self.unpickler.load()
            self.query_item = query_item

    def __getstate__(self):
        return (self.file_handler.name, self.offset_keys)

    def __setstate__(self, state):
        self.__init__(*state)

    def list_keys(self):
        return self.index.keys()

    def query_item(self, key):
        self.file_handler.seek(self.index[key])
        ret = self.unpickler.load()
        self.unpickler.memo.clear()
        return ret

    def optimize_query_order(self, dataset):
        if not self.offset_keys:
            def offset_of_key(key):
                return self.index[key]
        else:
            def offset_of_key(key):
                return key

        old_query = dataset.query

        def query(keys):
            keys.sort(key=offset_of_key)
            return old_query(keys)
        dataset.query = query


class DiffReason(Exception):
    pass


def _try_showing_pickling_button(dataset_to_update, dataset, path,
                                 check_first_n_items):
    try:
        from ipywidgets import Button, Layout
        from IPython import display
    except ImportError:
        pass
    else:
        button = Button(description="Regenerate {}".format(path),
                        layout=Layout(width="100%"))

        def rewrite_and_update(_):
            button.disabled = True
            desc = button.description
            button.description = desc + " (running)"
            dataset_to_update.file_handler.close()
            pickle_or_load(dataset, path, overwrite=True,
                           check_first_n_items=check_first_n_items,
                           show_overwrite_button=False)
            button.description = desc + " (finished)"
            sys.stderr.write("Dataset updated, do not forget to run the "
                             "cell again!\n")
        button.on_click(rewrite_and_update)
        display.display(button)


_open_pickles = weakref.WeakValueDictionary()


def _close(path):
    opened = _open_pickles.get(path, None)
    if opened is not None:
        opened.close()


@functools.wraps(open)
def _open_once(path, *args, **kwargs):
    _close(path)
    ret = _open_pickles[path] = open(path, *args, **kwargs)
    return ret


def pickle_or_load(dataset, path, check_first_n_items=1, overwrite=False,
                   show_overwrite_button=True):
    from io import IOBase
    if isinstance(path, IOBase):
        PickledDataset.create(dataset, path)
        return PickledDataset(path)
    was_existing = os.path.exists(path)
    if overwrite and was_existing:
        _close(path)
        os.remove(path)
        was_existing = False
    if not was_existing:
        file = None
        try:
            with _open_once(path, "wb") as file:
                PickledDataset.create(dataset, file)
        except BaseException as exc:  # catch ALL exceptions
            if file is not None:  # if the file has been created, it is partial
                os.remove(path)
            raise exc
    opened_dataset = PickledDataset(_open_once(path, "rb"))
    chunk = next(chunkify(dataset.list_keys(), check_first_n_items))
    reason = None
    for k in chunk:
        true_item = dataset.query_item(k)
        try:
            try:
                loaded_item = opened_dataset.query_item(k)
            except KeyError as exc:
                raise DiffReason("Pickled dataset does not contain key " +
                                 str(exc))
            equality = dataset.items_equality(true_item, loaded_item)
        except DiffReason as r:
            reason = r
            equality = False
        except Exception as e:
            sys.stderr.write("Warning: Could not check whether the dataset "
                             "pickled at {} was up to date, please implement "
                             "dataset.items_equality or item.__eq__.\n"
                             .format(path))
            raise e
        if not equality:
            if reason is not None:
                sys.stderr.write("Warning: Pickled dataset at {} seems to be "
                                 "out of date.\nReason: {}"
                                 "\n".format(path, str(reason)))
            else:
                sys.stderr.write("Warning: Pickled dataset at {} seems to be "
                                 "out of date. Or dataset.items_equality or "
                                 "item.__eq__ may be wrongly implemented."
                                 "\n".format(path))
            if not was_existing:
                sys.stderr.write("Since the dataset has just been created, it "
                                 "you may want to check the determinism of "
                                 "dataset.query_item.\n")
            elif show_overwrite_button:
                _try_showing_pickling_button(opened_dataset, dataset, path,
                                             check_first_n_items)
            break
    return opened_dataset


class DictDataset(Dataset):
    """Mostly an example for a simple in-memory dataset"""
    def __init__(self, dic):
        self.dic = dic

    def list_keys(self):
        return self.dic.keys()

    def query_item(self, key):
        return self.dic[key]


@warning(information="is deprecated and will most likely be removed in a near "
                     "future.")
class ParallelizeDataset(Dataset):
    """Since process pools are not efficient, use
    mlworkflow.SideRunner().yield_async(dataset.batches(...)) instead of
    wrapping your dataset in a ParallelizeDataset
    """
    # TODO: remove
    def __init__(self, dataset, pool, batches_ahead=1,
                 processes_per_batch=None):
        self.dataset = dataset
        self.pool = pool
        self.batches_ahead = batches_ahead
        self.processes_per_batch = (pool._processes
                                    if processes_per_batch is None
                                    else processes_per_batch)

    def list_keys(self):
        return self.dataset.list_keys()

    def query_item(self, key):
        return self.dataset.query_item(key)

    def query(self, keys, pool=None):
        if pool is None:
            pool = self.pool
        n = len(keys)
        Xs = [0]*n
        Ys = [0]*n
        results = pool.map(self.dataset.query_item, keys,
                           math.ceil(n / self.processes_per_batch))
        for i, result in enumerate(results):
            Xs[i], Ys[i] = result
        return np.array(Xs), np.array(Ys)

    def batches(self, keys, batch_size, in_advance=1, pool=None):
        if pool is None:
            pool = self.pool
        iterable = chunkify(keys, batch_size)
        pending = deque()
        items_per_process = math.ceil(batch_size/self.processes_per_batch)
        for i, key_chunk in zip(range(in_advance), iterable):
            pending.append(pool.map_async(self.dataset.query_item, key_chunk,
                                          items_per_process))
        for key_chunk in iterable:
            pending.append(pool.map_async(self.dataset.query_item, key_chunk,
                                          items_per_process))
            p = pending.popleft().get()
            n = len(p)
            Xs = [0]*n
            Ys = [0]*n
            for i, result in enumerate(p):
                Xs[i], Ys[i] = result
            yield np.array(Xs), np.array(Ys)
        for p in pending:
            p = p.get()
            n = len(p)
            Xs = [0]*n
            Ys = [0]*n
            for i, result in enumerate(p):
                Xs[i], Ys[i] = result
            yield np.array(Xs), np.array(Ys)

    def __del__(self):
        if self.pool is not None:
            self.pool.terminate()


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
