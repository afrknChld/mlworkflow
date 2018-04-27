from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

_no_value = object()


class Evaluable(metaclass=ABCMeta):
    @abstractmethod
    def eval(self, env):
        pass


def _transform_call_arg(value):
    if not no_dsl.dsl_enabled:
        return value

    if isinstance(value, str):
        if value.startswith("@"):
            return Ref(value[1:])

    return value


@contextmanager
def no_dsl():
    """Disables the DSL transformation when passing call arguments

    Without the DSL:
    >>> with no_dsl():
    ...     print(Environment(a=Call(print).with_args("compute a", "@b"),
    ...                       b=Call(print).with_args("compute b")))
    Environment(
        a=Call('builtins', 'print').with_args( 'compute a', '@b' ),
        b=Call('builtins', 'print').with_args( 'compute b' )
    )

    With the DSL again:
    >>> print(Environment(a=Call(print).with_args("compute a", "@b"),
    ...                   b=Call(print).with_args("compute b")))
    Environment(
        a=Call('builtins', 'print').with_args( 'compute a', Ref('b') ),
        b=Call('builtins', 'print').with_args( 'compute b' )
    )
    """
    _dsl = no_dsl.dsl_enabled
    no_dsl.dsl_enabled = False
    yield
    no_dsl.dsl_enabled = _dsl
no_dsl.dsl_enabled = True


class Call(Evaluable):
    """A malleable and picklable representation for a call.

    >>> call = Call(print).with_args("a", "b")(sep=",")
    >>> call.eval()
    a,b
    >>> call.with_args(..., "c", ...)(sep=" ").eval()
    a b c a b

    How to use it with pickle:

    >>> import pickle
    >>> a = Call(print)(sep=",").with_args("foo","bar")
    >>> s = pickle.dumps(a)     # This call is picklable
    >>> pickle.loads(s).eval()  # Restore and evaluate
    foo,bar
    """

    def __init__(self, callable, modattr=None, *, args=(), kwargs={}):
        if modattr is None:
            self.reference = (callable.__module__, callable.__name__)
        else:
            self.reference = (callable, modattr)
        self.args = args
        self.kwargs = kwargs

    def __reduce__(self):
        return _Call_v0, (self.reference, self.args, self.kwargs)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key0 = key[0]
            key1 = key[1] if len(key) == 2 else key[1:]
            return self.kwargs[key0][key1]
        return self.kwargs[key]

    def on(self, callable, modattr=None):
        return Call(callable, modattr, args=self.args, kwargs=self.kwargs)

    def with_args(self, *args):
        new_args = sum(((_transform_call_arg(arg),)
                        if arg is not Ellipsis else self.args
                        for arg in args), ())
        return Call(*self.reference, args=new_args, kwargs=self.kwargs)

    def __call__(self, **kwargs):
        new_kwargs = {**self.kwargs, **{k: _transform_call_arg(v)
                                        for k, v in kwargs.items()}}
        return Call(*self.reference, args=self.args, kwargs=new_kwargs)

    @property
    def resolved_target(self):
        import importlib
        call_target = getattr(self, "_call_target", _no_value)
        if call_target is not _no_value:
            return call_target
        module, attrs = self.reference
        current = importlib.import_module(module)
        for attr in attrs.split("."):
            current = getattr(current, attr)
        call_target = self._call_target = current
        return call_target

    def eval(self, env=None):
        args = [arg.eval(env) if isinstance(arg, Evaluable) else arg
                for arg in self.args]
        kwargs = {k: v.eval(env) if isinstance(v, Evaluable) else v
                  for k, v in self.kwargs.items()}
        return self.resolved_target(*args, **kwargs)

    def __eq__(self, other):
        return (self.reference == other.reference and
                self.args == other.args and
                self.kwargs == other.kwargs)

    def __repr__(self):
        kwargs = args = ""
        if self.args:
            args = ".with_args({})".format(", ".join(repr(arg)
                                                     for arg in self.args))
        if self.kwargs:
            kwargs = "({})".format(", ".join("{}={!r}".format(k, v)
                                             for k, v in self.kwargs.items()))
        return "Call{!r}{}{}".format(self.reference, args, kwargs)

    def __str__(self):
        s = ["Call{!r}".format(self.reference)]
        indentation = " "*2
        nl_indent = "\n{}".format(indentation)
        if self.kwargs:
            s.append("(")
            s.append(nl_indent)
            for k, v in self.kwargs.items():
                _v = str(v) if isinstance(v, Call) else repr(v)
                head = k+"="
                s.append(head)
                _nl_indent = nl_indent + " "*len(head)
                s.append(_nl_indent.join(_v.split("\n")))
                s.append(",\n" + indentation)
            s[-1] = s[-1][1:-1]  # remove comma and space
            s.append(")")
        if self.args:
            s.append(".with_args(")
            s.append(nl_indent)
            for arg in self.args:
                _v = str(arg) if isinstance(arg, Call) else repr(arg)
                s.append(nl_indent.join(_v.split("\n")))
                s.append(",\n" + indentation)
            s[-1] = s[-1][1:-1]  # remove comma and space
            s.append(")")
        return "".join(s)


class Ref(Evaluable):
    """A reference to a root element of a computation graph"""

    def __init__(self, name):
        self.name = name

    def __reduce__(self):
        return _Ref_v0, (self.name,)

    def eval(self, env):
        return env.run(self.name)

    def __repr__(self):
        return "Ref({!r})".format(self.name)


class Unique(Ref):
    """A reference to a root element of a computation graph for which
    computation won't use the cache"""

    def __reduce__(self):
        return _Unique_v0, (self.name,)

    def eval(self, env):
        return env[self.name].eval(env)

    def __repr__(self):
        return "Unique({!r})".format(self.name)


class Environment(dict):
    """A malleable and persistent representation for a computation graph

    >>> env = Environment(a=Call(print).with_args("compute a", "@b"),
    ...                   b=Call(print).with_args("compute b"))
    >>> env.run("a")
    compute b
    compute a None
    >>> env.run("b")
    >>> env.clean()
    >>> env.run("b")
    compute b
    >>> env.fused
    {'a': Call('builtins', 'print').with_args('compute a', Ref('b')),
     'b': None}
    >>> env.run(["a", "b"])
    compute a None
    [None, None]

    >>> import pickle
    >>> s = pickle.dumps(env)
    >>> pickle.loads(s).run(["a", "b"])
    compute b
    compute a None
    [None, None]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def __reduce__(self):
        return _Environment_v0, (dict(self),)

    def __setitem__(self, key, value):
        """
        >>> env = Environment()
        >>> env["a"] = "b"
        >>> env.update(b="c")
        >>> env.run(["a", "b"])
        ['b', 'c']
        """
        self.cache.pop(key, None)
        return super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        to_remove = {}  # empty dict on which we simulate the update
        to_remove.update(*args, **kwargs)
        for k in to_remove:
            self.cache.pop(k, None)
        return super().update(*args, **kwargs)

    def run(self, name):
        if isinstance(name, list):
            return [self.run(n) for n in name]
        value = self.cache.get(name, _no_value)
        if value is not _no_value:
            return value
        # Not in the cache, evaluate if Evaluable
        value = self[name]
        if isinstance(value, Evaluable):
            value = value.eval(self)
        self.cache[name] = value
        return value

    @property
    def fused(self):
        return Environment({**self, **self.cache})

    def clean(self):
        self.cache = {}

    def __str__(self):
        indentation = " "*2
        nl_indent = "\n"+indentation
        s = ["Environment("]
        if self:
            for k, v in self.items():
                s.append(nl_indent)
                head = "{}=".format(k)
                s.append(head)
                _nl_indent = nl_indent + " "*len(head)
                _v = str(v) if isinstance(v, (Call, Environment)) else repr(v)
                s.append(_nl_indent.join(_v.split("\n")))
                s.append(",")
            s[-1] = ""
            s.append("\n")
        s.append(")")
        return "".join(s)


# Versioning as those classes may well get pickled and evolve
def _Environment_v0(dic):
    return Environment(dic)


def _Call_v0(reference, args, kwargs):
    return Call(*reference, args=args, kwargs=kwargs)


def _Ref_v0(name):
    return Ref(name)


def _Unique_v0(name):
    return Unique(name)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
