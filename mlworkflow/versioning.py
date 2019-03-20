from mlworkflow.file_handling import _format_filename, find_files
from contextlib import contextmanager
import importlib
import builtins
import shutil
import os


@contextmanager
def imports(**redirections):
    def find_and_load_(name, import_):
        """Only way to monkey patch importlib.import_module consistently"""
        splits = name.split(".")
        target = redirections.get(splits[0], None)
        if target is not None:
            name = ".".join((target, *splits[1:]))
        return _find_and_load(name, import_)

    def import_(name, globals=None, locals=None, fromlist=(), level=0):
        req = name.split(".")
        target = redirections.get(req[0], None)
        if target is not None:
            name = ".".join((target, *req[1:]))
        else:
            return _import(name, globals, locals, fromlist, level)
        imp = _import(name, globals, locals, fromlist, level)
        # if fromlist, we receive the right object we can extract the fields from and it is OK
        # otherwise, we receive the root object and have to get to the one we want
        if not fromlist:
            ts = target.split(".")[1:]
            for t in ts:
                imp = getattr(imp, t)
        return imp
    
    _import = builtins.__import__
    _find_and_load = importlib._bootstrap._find_and_load
    try:
        builtins.__import__ = import_
        importlib._bootstrap._find_and_load = find_and_load_
        yield
    finally:
        builtins.__import__ = _import
        importlib._bootstrap._find_and_load = _find_and_load


class TimeCapsule:
    def __init__(self, base, dirname, files="*"):
        self.base = os.path.dirname(base) if base.endswith(".py") else base
        self.target = _format_filename(dirname)
        self.base_target = os.path.join(self.base, self.target)
        self.files = files.split(",")

    def build(self):
        self._create_target()
        self._copy_files()
        return self.base_target

    def __enter__(self):
        self._create_target()
        self._copy_files()
        return self.base_target

    def __exit__(self, type, value, traceback):
        self._remove_if_no_change()

    def _compute_dirs(self, files):
        dirs = set()
        for file in files:
            dirs.add(os.path.dirname(file))
        return dirs

    def _create_target(self):
        os.makedirs(self.base_target)

    def _copy_files(self):
        files = find_files(self.files, base_dir=self.base)
        # Create dirs
        for dir_ in sorted(self._compute_dirs(files)):
            os.makedirs(os.path.join(self.base_target, dir_), exist_ok=True)
        # Copy files into them
        for file in files:
            shutil.copyfile(os.path.join(self.base, file), os.path.join(self.base_target, file))
        self.copied_files = files

    def _remove_if_no_change(self):
        files = set(find_files("**", base_dir=self.base_target))
        old_files = set(self.copied_files)
        files.difference_update(old_files)
        if not files:
            for file in self.copied_files:
                os.remove(os.path.join(self.base_target, file))
            for dir_ in sorted(self._compute_dirs(self.copied_files), reverse=True):
                os.rmdir(os.path.join(self.base_target, dir_))
