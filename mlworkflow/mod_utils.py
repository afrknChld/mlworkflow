from contextlib import contextmanager
from importlib import import_module
from datetime import datetime
from types import ModuleType

import linecache
import textwrap
import inspect
import difflib
import time
import sys
import re

_no_value = object()
_leading_whitespace_re = re.compile('^([ ]*)(?:[^ \n])')


class ModuleSaver:
    """
    >>> mod = ModuleSaver("test", "print('evaluating');a=1;b=a+1")
    >>> mod.a
    evaluating
    1
    >>> mod.b
    2
    >>> mod.c
    Traceback (most recent call last):
        ...
    AttributeError: module 'test_...' has no attribute 'c'
    >>> with mod.as_base():
    ...     import test
    ...     print(test is mod.module)
    True
    """

    def __init__(self, base_name, source=None, *,  uname=None):
        if source is None:
            if isinstance(base_name, str):
                module = import_module(base_name)
            else:
                module = base_name
                assert isinstance(module, ModuleType)
            base_name = module.__name__
            source = inspect.getsource(module)
        if uname is None:
            uname = "{}_{}".format(base_name,
                                   datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.base_name = base_name
        self.uname = uname
        self.source = source
        self._module = None

    @property
    def module(self):
        if self._module is None:
            self._module = ModuleType(self.base_name)
            code_cache.cache(self.source, self.uname)
            code = compile(self.source, self.uname, "exec")
            exec(code, self._module.__dict__)
        return self._module

    def __getattr__(self, name):
        return getattr(self.module, name)

    @contextmanager
    def as_base(self):
        base_name = self.base_name
        base_module = sys.modules.get(base_name, _no_value)
        sys.modules[base_name] = self.module
        try:
            yield
        finally:
            if base_module is _no_value:
                del sys.modules[base_name]
            else:
                sys.modules[base_name] = base_module

    def diff(self, mode="unified", base_is_target=True, colored=False):
        def _color(line):
            if line.startswith("+"):
                return "\x1b[32m"+line+"\x1b[0m"
            if line.startswith("-"):
                return "\x1b[31m"+line+"\x1b[0m"
            return line

        fname = self.uname
        flines = self.source.splitlines()
        tname = self.base_name
        tlines = inspect.getsource(import_module(self.base_name)).splitlines()
        if not base_is_target:
            fname, tname = tname, fname
            flines, tlines = tlines, flines

        if mode == "unified":
            diff = difflib.unified_diff(flines, tlines, lineterm="",
                                        fromfile=fname,
                                        tofile=tname)
            if colored:
                diff = (_color(d) for d in diff)
            return "\n".join(diff)
        raise Exception("{!r} is not a valid value for 'mode'".format(mode))

    def __repr__(self):
        return "ModuleSaver({!r})".format(self.uname)
    
    def __reduce__(self):
        return ModuleSaver._v0, (self.base_name, self.source,
                                 self.uname)

    @staticmethod
    def _v0(base_name, source, uname):
        return ModuleSaver(base_name, source, uname=uname)


class _CodeCache:
    """An object that caches Python code in order to be able to access the
    code of replayed modules in tracebacks.
    Mostly copy-pasted from IPython.core.compilerop.CachingCompiler
    """

    def __init__(self):
        if not hasattr(linecache, '_mlwf_cache'):
            linecache._mlwf_cache = {}
        if not hasattr(linecache, '_mlwf_checkcache_ori'):
            linecache._mlwf_checkcache_ori = linecache.checkcache
        # Now, we must monkeypatch the linecache directly so that parts of the
        # stdlib that call it outside our control go through our codepath
        # (otherwise we'd lose our tracebacks).
        linecache.checkcache = _check_linecache_mlworkflow
        
    def cache(self, code, name):
        entry = (len(code), time.time(),
                 [line+'\n' for line in code.splitlines()], name)
        linecache.cache[name] = entry
        linecache._mlwf_cache[name] = entry
        return name


def _check_linecache_mlworkflow(*args):
    """Call linecache.checkcache() safely protecting our cached values."""
    # First call the original checkcache as intended
    linecache._mlwf_checkcache_ori(*args)
    # Then, update back the cache with our data, so that tracebacks related
    # to our compiled codes can be produced.
    linecache.cache.update(linecache._mlwf_cache)


code_cache = _CodeCache()


_pattern_filters = {"*":  r"[^\.]*",
                    "**": r".*"
                    }


def _select_filter(match):
    match = match.group()
    return _pattern_filters.get(match, match)


class ModuleCollection:
    def __init__(self, *, modules=None, pattern=None):
        self.modules = []

        if modules:
            for module in modules:
                if isinstance(module, ModuleSaver):
                    self.modules.append(module)
                else:
                    self.modules.append(ModuleSaver(module))

        if pattern is not None:
            pattern = re.sub(r"\*+", _select_filter, pattern.replace(".", r"\."))
            pattern = re.compile("^{}$".format(pattern))
            for modname in sys.modules:
                if pattern.match(modname) is not None:
                    self.modules.append(ModuleSaver(sys.modules[modname]))

    def __enter__(self):
        self._ctx_managers = [m.as_base() for m in self.modules]
        for ctx_manager in self._ctx_managers:
            ctx_manager.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        handled_exception = None
        for ctx_manager in self._ctx_managers[::-1]:
            if ctx_manager.__exit__(exc_type, exc_value, traceback):
                handled_exception = True
                exc_type, exc_value, traceback = None, None, None
        return handled_exception

    def __repr__(self):
        return "ModuleCollection({!r})".format(self.modules)

    def __reduce__(self):
        return ModuleSet._v0, (self.modules,)

    @staticmethod
    def _v0(modules):
        module_set = ModuleSet(modules=modules)
        return module_set


try:
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic
except ImportError:
    pass
else:
    _ip = get_ipython()
    if _ip is not None:
        @register_cell_magic
        def virtual_module(line, cell):
            name, *flags = line.split()
            mod = ModuleSaver(name, cell)
            sys.modules[name] = _ip.user_global_ns[name] = mod.module


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
