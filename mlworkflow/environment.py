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
        return Call._v0, (self.reference, self.args, self.kwargs)

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
        if not isinstance(other, Call):
            return False
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

    @staticmethod
    def _v0(reference, args, kwargs):
        return Call(*reference, args=args, kwargs=kwargs)


class Ref(Evaluable):
    """A reference to a root element of a computation graph"""

    def __init__(self, name, *keys):
        self.name = name
        self.keys = keys

    def __reduce__(self):
        return Ref._v0, (self.name,) + self.keys

    def eval(self, env):
        item = env.run(self.name)
        for k in self.keys:
            item = item[k]
        return item

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return False
        return self.name == other.name and self.keys == other.keys

    def __repr__(self):
        return "Ref({})".format(", ".join(repr(k)
                                          for k in ((self.name,) + self.keys)))

    @staticmethod
    def _v0(name):
        return Ref(name)


class Exec(Evaluable, dict):
    """The locals generated by running some Python code

    >>> env = Environment(someValues=Exec('''
    ... a = 1
    ... b = 2
    ... env = _env
    ... _z = "not exported"
    ... print("Code finished running")
    ... '''))
    >>> env.run("someValues") == {"a": 1, "b": 2, "env": env}
    Code finished running
    True

    But we can also use them as simple in-place functions
    >>> env["x"] = Exec('''
    ... someValues = _env.run("someValues")
    ... a, b = someValues["a"], someValues["b"]
    ... c = a + b
    ... _set_result(c)
    ... ''')
    >>> env.run("x")
    3

    >>> env["y"] = Exec('''
    ... #@export d
    ... d = someValues["a"] + someValues["b"]
    ... e = d + someValues["a"] + someValues["b"]
    ... ''', cross_refs=True)
    >>> env.run("y") == {"d": 3}
    True
    >>> env.run("d")
    3
    """
    def __init__(self, code, cross_refs=False):
        self.code = code
        self.cross_refs = cross_refs

    def __reduce__(self):
        return Exec._v0, (self.code, self.cross_refs)

    def eval(self, env=None):
        _result = _no_value
        def set_result(result):
            nonlocal _result
            _result = result
        locs = Exec._Locals(env, _env=env, _set_result=set_result)
        exec(self.code, locs)
        if _result is not _no_value:
            return _result
        # If there are @export comments, use them, otherwise, all non
        # _-beginning variables are exported
        exported = set(sum((line.split()[1:]
                            for line in self.code.split("\n")
                            if line.startswith("#@export ")), []))
        if not exported:
            exported = set(k for k in locs if not k.startswith("_"))
        # Only retain exported variables
        locs = {k:v for k, v in locs.items() if k in exported}
        if self.cross_refs:
            assert env[env.current] == self, ("Cannot run a non root Exec "
                                              "with cross_refs option")
            for n in locs:
                ref = Ref(env.current, n)
                current_item = env.get(n, _no_value)
                assert current_item is _no_value or ref == current_item, \
                    ("Variable {!r} was already defined in field {!r}." 
                     .format(n, current_item.name))
                env[n] = ref
        return locs

    class _Locals(dict):
        def __init__(self, env, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.env = env

        def __getitem__(self, key):
            value = super().get(key, _no_value)
            if value is not _no_value:
                return value
            if key in self.env:
                return self.env.run(key)
            else:
                raise KeyError(key)

    def __str__(self):
        return ("Exec('''\n{}\n''', cross_refs={!r})"
                .format(self.code.replace("'''", r'\'\'\''), self.cross_refs))

    def __repr__(self):
        return "Exec({!r}, cross_refs={!r})".format(self.code, self.cross_refs)

    @staticmethod
    def _v0(code, cross_refs=False):
        return Exec(code, cross_refs=cross_refs)
            

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

    >>> env.run("_env") is env
    True
    """

    current = _no_value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = dict(_env=self)

    def __reduce__(self):
        return Environment._v0, (dict(self),)

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
        if value is _no_value:
            # Not in the cache, evaluate if Evaluable
            value = self[name]
            if isinstance(value, Evaluable):
                _current = self.current
                self.current = name
                value = value.eval(self)
                self.current = _current
            self.cache[name] = value
        return value

    @property
    def fused(self):
        cache_without_self = {k: v
                              for k, v in self.cache.items()
                              if k != "_env" or "_env" in self
                             }
        return Environment({**self, **cache_without_self})

    def clean(self):
        if self.get("_env", _no_value) is _no_value:
            self.cache = dict(_env=self)
        else:
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
                _v = str(v) \
                    if isinstance(v, (Call, Environment, Exec)) \
                    else repr(v)
                s.append(_nl_indent.join(_v.split("\n")))
                s.append(",")
            s[-1] = ""
            s.append("\n")
        s.append(")")
        return "".join(s)

    @staticmethod
    def _v0(dic):
        return Environment(dic)


try:
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic
except ImportError:
    pass
else:
    _ip = get_ipython()
    if _ip is not None:
        @register_cell_magic
        def with_env(line, cell):
            env, name, *flags = line.split(" ")
            env = _ip.user_global_ns[env]
            env[name] = Exec(cell, cross_refs=True)
            res = env.run(name)
            if "silent" not in flags:
                return res


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
