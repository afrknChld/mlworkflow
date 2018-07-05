from contextlib import contextmanager
from collections import ChainMap
import numpy as np
import functools
import sys


@contextmanager
def _seed(random, seed):
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


_no_value = object()


class ctx_or:
    def __init__(self, default_value):
        self.default_value = default_value
    def __repr__(self):
        return "ctx_or({!r})".format(self.default_value)


def bind(f, **bkwargs):
    """Creates a new function with added kwargs"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = {**bkwargs, **kwargs}
        return f(*args, **kwargs)
    return wrapper


def bindable(f):
    """Add a bind(**kwargs) attribute to create a new function with
    added default kwargs.
    """
    def bind(**bkwargs):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            kwargs = {**bkwargs, **kwargs}
            return f(*args, **kwargs)
        return wrapper
    f.bind = bind
    return f


def kwonly_from_ctx(f):
    """Wraps a function so that it can unpack arguments from a ctx dictionary.
    Those arguments may only be keywords and have no default value.

    Only kwonly arguments without default value are replaced by the context, or
    those with a ctx_or(v) value.
    """
    import inspect
    argspec = inspect.getfullargspec(f)
    kwonlyargs = argspec.kwonlyargs
    kwonlydefaults = argspec.kwonlydefaults
    if kwonlyargs is None:
        kwonlyargs = []
    fillable = set(kwonlyargs)
    if kwonlydefaults is not None:
        # Do not fill arguments with default value, except if it is a ctx_or
        not_to_fill = (k for k, v in kwonlydefaults.items()
                       if not isinstance(v, ctx_or))
        fillable.difference_update(not_to_fill)
    else:
        kwonlydefaults = {}
    fillable.discard("ctx")  # the ctx argument is handled separately

    @functools.wraps(f)
    def wrapper(*args, ctx=_no_value, **kwargs):
        if ctx is _no_value:
            return f(*args, **kwargs)
        elif isinstance(ctx, (list, tuple)):
            ctx = ChainMap(*ctx[::-1])  # Last added dict has highest priority
        if "ctx" in kwonlyargs:
            kwargs["ctx"] = ctx
        for name in fillable:
            # If the parameter is provided, do not fill
            if kwargs.get(name, _no_value) is not _no_value:
                continue
            # Otherwise, fill if it is provided by ctx.
            from_ctx = ctx.get(name, _no_value)
            if from_ctx is not _no_value:
                kwargs[name] = from_ctx
            # Otherwise, provide its default value. It is fillable, a ctx_or
            from_default = kwonlydefaults.get(name, _no_value)
            if from_default is not _no_value:
                kwargs[name] = from_default.default_value
        return f(*args, **kwargs)
    return _attach_ops(wrapper)


def _wrap(ctx):
    if isinstance(ctx, (tuple, list)):
        return ctx
    return (ctx,)


def _attach_ops(f):
    def bind_ctx(*contexts, lock=False):
        @functools.wraps(f)
        def wrapper(*args, ctx=_no_value, **kwargs):
            if ctx is _no_value:
                ctx = contexts
            else:
                assert not lock
                ctx = contexts + _wrap(ctx)
            return f(*args, ctx=ctx, **kwargs)
        return _attach_ops(wrapper)
    f.bind_ctx = bind_ctx
    return f


@functools.wraps(exec)
def _exec(source, level=0, custom_globals=None):
    frame = sys._getframe(level+1)
    if custom_globals is not None:
        frame.f_globals.update(custom_globals)
    return exec(source, frame.f_locals, frame.f_globals)


class DictObject(dict):
    __slots__ = ()

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, value):
        return super().__setitem__(key, value)

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

    def __copy__(self):
        copy = self.__class__.__new__(self.__class__)
        for k, v in items:
            copy[k] = v
        return copy
    copy = __copy__
    
    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        copy = self.__class__.__new__(self.__class__)
        for k, v in items:
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
        # We bypass __init__ because new fields mostly for the reason that new
        # fields may have been added while __init__ may not expect them.
        target = klass.__new__(klass)
        target.update(items)
        return target


class ListQueriableDict(dict):
    def __getitem__(self, key):
        sup = super()
        if isinstance(key, list):
            return [sup.__getitem__(k) for k in key]
        return sup.__getitem__(key)

    def __setitem__(self, key, value):
        sup = super()
        if isinstance(key, list):
            for k, v in zip(key, value):
                sup.__setitem__(k, v)
        sup.__setitem__(key, value)
