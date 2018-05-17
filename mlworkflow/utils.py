import functools
import sys


_no_value = object()


class ctx_or:
    def __init__(self, default_value):
        self.default_value = default_value
    def __repr__(self):
        return "ctx_or({!r})".format(self.default_value)


def kwonly_from_ctx(f):
    """Wraps a function so that it can unpack arguments from a ctx dictionary.
    Those arguments may only be keywords and have no default value.
    """
    import inspect
    argspec = inspect.getfullargspec(f)
    kwonlyargs = argspec.kwonlyargs
    kwonlydefaults = argspec.kwonlydefaults
    if kwonlyargs is None:
        kwonlyargs = []
    fillable = set(kwonlyargs)
    if kwonlydefaults is not None:
        not_to_fill = (k for k, v in kwonlydefaults.items()
                       if not isinstance(v, ctx_or))
        fillable.difference_update(not_to_fill)
    else:
        kwonlydefaults = {}
    fillable.discard("ctx")  # handled separately

    @functools.wraps(f)
    def wrapper(*args, ctx=_no_value, **kwargs):
        if ctx is _no_value:
            return f(*args, **kwargs)
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
            # Otherwise, provide its default value.
            from_default = kwonlydefaults.get(name, _no_value)
            if from_default is not _no_value:
                kwargs[name] = from_default.default_value
        return f(*args, **kwargs)
    return wrapper


def bind_ctx(f, ctx):
    return functools.partial(f, ctx=ctx)


def chain_ctx(*args):
    from collections import ChainMap
    ctx = ChainMap(*args)
    return ctx


@functools.wraps(exec)
def _exec(source, level=0, custom_globals=None):
    frame = sys._getframe(level+1)
    if custom_globals is not None:
        frame.f_globals.update(custom_globals)
    return exec(source, frame.f_locals, frame.f_globals)
