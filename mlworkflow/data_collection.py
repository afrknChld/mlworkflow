from itertools import takewhile
import datetime
import pickle
import os
import re


_pattern_filters = {"*":  r"[^{}]*".format(os.sep),
                    "**": r".*"
                    }


def _select_filter(match):
    match = match.group()
    return _pattern_filters.get(match, match)


def find_files(pattern="*.dcp", show_hidden=False):
    """Simply matches files following the provided pattern (one directory only)
    """
    sections = pattern.split("/")
    root = tuple(takewhile(lambda section: "*" not in section, sections[:-1]))
    pattern = sections[len(root):]

    pattern = os.sep.join(pattern)
    root = os.sep.join(root) if root else "./"
    _recursive = "**" in pattern

    pattern = re.sub(r"\**", _select_filter, pattern.replace(".", r"\."))
    pattern = re.compile("^{}$".format(pattern))

    lst = []
    prefix_length = len(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = dirpath[prefix_length:]
        if dirpath.startswith(".") and not show_hidden:
            continue
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            match = pattern.match(path)
            if match is not None:
                lst.append(os.path.join(root, path))
        if not _recursive:
            break
    lst.sort()
    return lst


def _create_file(filename):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
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


class _ExternalRetriever:
    def __init__(self, namer, dirname):
        self.namer = namer
        self.dirname = dirname

    def __getitem__(self, key):
        name = self.namer[key]
        return self.resolve(name)

    def resolve(self, name):
        if isinstance(name, list):
            return [self.resolve(n) for n in name]
        with open(os.path.join(self.dirname, name), "rb") as file:
            return pickle.load(file)


class _CheckPointFileWrapper(list):
    """Allows lists of _CheckPointWrapper and selection using slices"""
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

    @property
    def external(self):
        assert self.filename is not None
        return _ExternalRetriever(self, os.path.dirname(self.filename))


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
    def create_with_parent(filename="{}.dcp", parent=None, *, slice=None,
                           fields=None):
        assert parent is not None
        old = DataCollection.load_file(parent, slice=slice, fields=fields)
        data = DataCollection(filename)
        if data.filename is not None:
            relpath = os.path.relpath(os.path.dirname(parent),
                                      os.path.dirname(data.filename))
            data["_parent"] = os.path.join(relpath, os.path.basename(parent))
        else:
            data["_parent"] = parent
        return data, old

    @staticmethod
    def add_metadata(filename, dic):
        assert isinstance(dic, dict), ("metadata must take the form of a "
                                       "dictionary")
        with open(filename+"_", "ab") as file:
            pickle.Pickler(file).dump(dic)

    @staticmethod
    def get_metadata(filename):
        metadata = {}
        try:
            with open(filename+"_", "rb") as file:
                unpickler = pickle.Unpickler(file)
                try:
                    while True:
                        obj = unpickler.load()
                        metadata.update(obj)
                except EOFError:
                    pass
        except FileNotFoundError:
            pass
        return metadata

    @staticmethod
    def load_file(filename, *, slice=None, fields=None):
        data = []
        with open(filename, "rb") as file:
            unpickler = pickle.Unpickler(file)
            try:
                while True:
                    obj = unpickler.load()
                    if fields is not None:
                        obj = {k: v for k, v in obj.items() if k in fields}
                    obj = _CheckPointWrapper(obj)
                    data.append(obj)
            except EOFError:
                pass
        if slice is not None:
            data = data[slice]
        return _CheckPointFileWrapper(data, filename=filename)

    @staticmethod
    def load_files(filenames, *, slice=None, fields=None):
        library = {}
        for filename in filenames:
            library[filename] = DataCollection.load_file(filename, slice=slice,
                                                         fields=fields)
        return _CheckPointLibraryWrapper(library)

    def __init__(self, filename="{}.dcp"):
        super().__init__()
        if filename is not None:
            filename = _format_filename(filename)
        self.history = _CheckPointFileWrapper([], filename=filename)
        self.filename = filename
        if self.filename is not None:
            self.dir = os.path.dirname(self.filename)
            self.base = os.path.basename(self.filename)
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
    def iteration(self):
        return len(self.history)

    def save_external(self, key, value, may_skip=False):
        if self.filename is None:
            assert may_skip, ("No filename is provided. Please either provide "
                              "one or allow skipping")
            return
        new_base = "{}_{}_{}".format(self.base, self.iteration, key)
        new_filename = os.path.join(self.dir, new_base)
        with _create_file(new_filename) as file:
            pickle.dump(value, file)
        self[key] = new_base
        return new_filename

    @property
    def history_(self):
        return _CheckPointFileWrapper(self.history+[_CheckPointWrapper(self)],
                                      filename=self.filename)

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


class DataCollection_Legacy0(dict):
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
    def create_with_parent(filename="{}.dcp", parent=None, *, slice=None,
                           fields=None):
        assert parent is not None
        old = DataCollection.load_file(parent, slice=slice, fields=fields)
        data = DataCollection(filename)
        if data.filename is not None:
            relpath = os.path.relpath(os.path.dirname(parent),
                                      os.path.dirname(data.filename))
            data["_parent"] = os.path.join(relpath, os.path.basename(parent))
        else:
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
            data = _CheckPointFileWrapper(data, filename=filename)
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
        if filename is not None:
            filename = _format_filename(filename)
        self.history = _CheckPointFileWrapper([], filename=filename)
        self.filename = filename
        if self.filename is not None:
            self.dir = os.path.dirname(self.filename)
            self.base = os.path.basename(self.filename)
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
    def iteration(self):
        return len(self.history)

    def save_external(self, key, value, may_skip=False):
        if self.filename is None:
            assert may_skip, ("No filename is provided. Please either provide "
                              "one or allow skipping")
            return
        new_base = "{}_{}_{}".format(self.base, self.iteration, key)
        new_filename = os.path.join(self.dir, new_base)
        with _create_file(new_filename) as file:
            pickle.dump(value, file)
        self[key] = new_base
        return new_filename

    @property
    def history_(self):
        return _CheckPointFileWrapper(self.history+[_CheckPointWrapper(self)],
                                      filename=self.filename)

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
