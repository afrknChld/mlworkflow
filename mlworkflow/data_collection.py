import datetime
import pickle
import os
import re


def find_files(pattern="{}.dcp"):
    """Simply matches files following the provided pattern (one directory only)
    """
    pattern = pattern.split(os.sep)
    directory, pattern = os.path.join(*pattern[:-1]) \
        if len(pattern) > 1 else ".", pattern[-1]
    pattern = re.compile("^" + pattern.replace(".", r"\.")
                                      .replace("{}", ".*") + "$")
    filenames = next(os.walk(directory))[2]
    lst = []
    for filename in filenames:
        match = pattern.match(filename)
        if match is not None:
            lst.append(os.path.join(directory, filename))
    lst.sort()
    return lst


def _create_file(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    file = open(filename, "wb")
    return file


def _format_filename(filename):
    return filename.format(datetime.datetime.now()
                           .strftime("%Y%m%d_%H%M%S"))


class _CheckPointWrapper(dict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return super().get(key.start, key.stop)
        if isinstance(key, list):
            return [self[k] for k in key]
        return super().__getitem__(key)


class _CheckPointFileWrapper(list):
    """Allows lists of _CheckPointWrapper and selection using slices"""
    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2, ("Key tuple must be of length 2,"
                                   "got {!r}".format(key))
            key0 = key[0]
            key1 = key[1]
            if isinstance(key0, slice):
                return [l[key1] for l in super().__getitem__(key0)]
            return super().__getitem__(key0)[key1]
        return super().__getitem__(key)

    def write_as(self, file="{}.dcp"):
        if isinstance(file, str):
            file = _format_filename(file)
            with _create_file(file) as f:
                self.write_as(f)
            return file

        pickler = pickle.Pickler(file)
        for checkpoint_wrapper in self:
            checkpoint = dict(checkpoint_wrapper)
            pickler.dump(checkpoint)
        return file


class _CheckPointLibraryWrapper(dict):
    """Allows a dict of 3rd level to be sliced and to propagate indexing"""
    def __init__(self, library):
        super().__init__(library)
        self._keys = list(self.keys())

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 3, ("Key tuple must be of length 3, got "
                                   "{!r}".format(key))
            key0 = key[0]
            key1 = key[1:]
            if isinstance(key0, (int, slice)):
                key0 = self._keys[key0]
            if isinstance(key0, list):
                return [super(_CheckPointLibraryWrapper, self)
                        .__getitem__(k)[key1]
                        for k in key0]
            return super().__getitem__(key0)[key1]
        if isinstance(key, int):
            key = self._keys[key]
        return super().__getitem__(key)


class _MetaData:  # TODO: Remove, breaks backward compatibility
    def __init__(self, data):
        self.data = data


class DataCollection(dict):
    """A class for recording experimental results

    >>> data = DataCollection(None) # No output file
    >>> for i in range(10):
    ...    data["iteration"] = i
    ...    data["error"] = 1/(10**i)
    ...    data.checkpoint()
    {...}
    >>> data.history[:,"iteration"]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> data.history[-2:, ["iteration", "error"]]
    [[8, 1e-08], [9, 1e-09]]
    """

    @staticmethod
    def load_with_parent(filename="{}.dcp", parent=None, *, slice=None,
                         fields=None):
        assert parent is not None
        old = DataCollection.load_file(parent, slice=slice, fields=fields)
        data = DataCollection(filename)
        data["_parent"] = parent
        return data, old

    @staticmethod
    def add_metadata(filename, dic):
        assert isinstance(dic, dict), ("metadata must take the form of a "
                                       "dictionary")
        with open(filename, "ab") as file:
            pickle.Pickler(file).dump(("metadata", dic))

    @staticmethod
    def get_metadata(filename):
        return DataCollection.get_data_metadata(filename, data=False,
                                                metadata=True)[1]

    @staticmethod
    def load_file(filename, *, slice=None, fields=None):
        return DataCollection.get_data_metadata(filename, data=True,
                                                metadata=False, slice=slice,
                                                fields=fields)[0]

    @staticmethod
    def get_data_metadata(filename, data=True, metadata=True, *,
                          slice=None, fields=None):
        assert data or metadata
        data = [] if data else None
        metadata = {} if metadata else None
        with open(filename, "rb") as file:
            unpickler = pickle.Unpickler(file)
            try:
                while True:
                    obj = unpickler.load()
                    is_metadata = False
                    if isinstance(obj, tuple) and obj[0] == "metadata":
                        is_metadata = True
                        obj = obj[1]
                    # TODO: Remove, breaks backward compatibility
                    elif isinstance(obj, _MetaData):
                        is_metadata = True
                        obj = obj.data
                    if is_metadata and metadata is not None:
                        metadata.update(obj)
                    elif not is_metadata and data is not None:
                        if fields is not None:
                            obj = {k: v for k, v in obj.items() if k in fields}
                        obj = _CheckPointWrapper(obj)
                        data.append(obj)
            except EOFError:
                pass
        if slice is not None:
            data = data[slice]
        if data is not None:
            data = _CheckPointFileWrapper(data)
        return data, metadata

    @staticmethod
    def load_files(filenames, *, slice=None, fields=None):
        library = {}
        for filename in filenames:
            library[filename] = DataCollection.load_file(filename, slice=slice,
                                                         fields=fields)
        return _CheckPointLibraryWrapper(library)

    def __init__(self, filename="{}.dcp"):
        super().__init__()
        self.history = _CheckPointFileWrapper([])
        if filename is not None:
            filename = _format_filename(filename)
        self.filename = filename
        self.file = None
        self.pickler = None

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.__getitem__(k) for k in key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            assert len(key) == len(value)
            for k, v in zip(key, value):
                self.__setitem__(k, v)
            return
        super().__setitem__(key, value)

    @property
    def history_(self):
        return _CheckPointFileWrapper(self.history+[_CheckPointWrapper(self)])

    def _checkpoint(self, checkpoint):
        self.history.append(_CheckPointWrapper(checkpoint))
        if self.filename is not None:
            if self.file is None:
                self.file = _create_file(self.filename)
                self.pickler = pickle.Pickler(self.file)
            self.pickler.dump(checkpoint)

    def checkpoint(self, *names):
        if not names:
            names = self.keys()
        checkpoint = {name: self[name]
                      for name in names}
        self._checkpoint(checkpoint)
        return checkpoint

    def close(self):
        if self.file is not None:
            self.file.close()


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
