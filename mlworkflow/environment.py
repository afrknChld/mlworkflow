from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from collections import defaultdict
import re

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


def _resolve_attr(obj, path, *, first_accessor=None):
    split = re.split(r"([\.@>])", path)
    accessors = split[1::2]
    keys = split[::2]
    if first_accessor is not None:
        accessors.insert(0, first_accessor)
    assert len(accessors) == len(keys)
    for acc, key in zip(accessors, keys):
        if acc == ".":
            obj = getattr(obj, key)
        elif acc == "@":
            obj = obj[key]
        elif acc == ">":
            obj = obj.run(key)
    return obj


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


class GlobalRef(Evaluable):
    def __init__(self, module, path):
        self.module = module
        self.path = path

    def __reduce__(self):
        return GlobalRef._v0, (self.module, self.path,)
    
    def eval(self, env=None):
        import importlib
        module = importlib.import_module(self.module)
        return _resolve_attr(module, self.path, first_accessor=".")

    def __eq__(self, other):
        if not isinstance(other, GlobalRef):
            return False
        return self.module == other.module and self.path == other.path

    def __repr__(self):
        return "GlobalRef({!r}, {!r})".format(self.module, self.path)

    @staticmethod
    def _v0(module, path):
        return GlobalRef(module, path)


class Ref(Evaluable):
    """A reference to a root element of a computation graph"""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Ref._v0, (self.path,)

    def eval(self, env):
        return _resolve_attr(env, self.path, first_accessor=">")

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return False
        return self.path == other.path

    def __repr__(self):
        return "Ref({!r})".format(self.path)

    @staticmethod
    def _v0(path):
        return Ref(path)


def ListFromArgs(*args):
    return list(args)


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

    def __init__(self, callable, modattr=None, *, args=(), kwargs={},
                 partial=False):
        if modattr is None:
            if isinstance(callable, Evaluable):
                self.reference = callable
            else:
                self.reference = GlobalRef(callable.__module__,
                                           callable.__qualname__)
        else:
            self.reference = GlobalRef(callable, modattr)
        self.args = args
        self.kwargs = kwargs
        self._partial = partial

    @staticmethod
    def _new(callable, modattr=None, *, args, kwargs, partial):
        """Internal variant of Call() enforcing all the parameters to have
        so it is easier to check that internal call creation respects clones
        all the fields it should"""
        return Call(callable, modattr, args=args, kwargs=kwargs,
                    partial=partial)

    def __reduce__(self):
        return Call._v1, (self.reference, self.args, self.kwargs,
                          self._partial)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key0 = key[0]
            key1 = key[1] if len(key) == 2 else key[1:]
            return self.kwargs[key0][key1]
        return self.kwargs[key]

    def on(self, callable, modattr=None):
        return Call._new(callable, modattr, args=self.args, kwargs=self.kwargs,
                         partial=self._partial)

    def with_args(self, *args):
        new_args = sum(((_transform_call_arg(arg),)
                        if arg is not Ellipsis else self.args
                        for arg in args), ())
        return Call._new(self.reference, args=new_args, kwargs=self.kwargs,
                         partial=self._partial)

    def __call__(self, **kwargs):
        new_kwargs = {**self.kwargs, **{k: _transform_call_arg(v)
                                        for k, v in kwargs.items()}}
        return Call._new(self.reference, args=self.args, kwargs=new_kwargs,
                         partial=self._partial)

    def eval(self, env=None):
        ref, args, kwargs = self._eval_rak(env)
        if self._partial:
            return lambda *_args, **_kwargs: ref(*args, *_args,
                                                 **kwargs, **_kwargs)
        return ref(*args, **kwargs)

    def _eval_rak(self, env):
        """Evaluates reference, arguments and keyword arguments recursively"""
        ref = self.reference.eval(env)
        args = [arg.eval(env) if isinstance(arg, Evaluable) else arg
                for arg in self.args]
        kwargs = {k: v.eval(env) if isinstance(v, Evaluable) else v
                  for k, v in self.kwargs.items()}
        return ref, args, kwargs

    def __eq__(self, other):
        if not isinstance(other, Call):
            return False
        return (self.reference == other.reference and
                self.args == other.args and
                self.kwargs == other.kwargs and
                self._partial == other._partial)

    def _format_ref(self):
        if isinstance(self.reference, GlobalRef):
            return "{!r}, {!r}".format(self.reference.module,
                                       self.reference.path)
        else:
            return repr(self.reference)

    def __repr__(self):
        kwargs = args = partial = ""
        if self.args:
            args = ".with_args({})".format(", ".join(repr(arg)
                                                     for arg in self.args))
        if self.kwargs:
            kwargs = "({})".format(", ".join("{}={!r}".format(k, v)
                                             for k, v in self.kwargs.items()))
        if self._partial:
            partial = ".partial()"
        return "Call({}){}{}{}".format(self._format_ref(),
                                       args, kwargs, partial)

    def __str__(self):
        s = ["Call({})".format(self._format_ref())]
        indentation = " "*3
        nl_indent = "\n{}".format(indentation)
        if self.kwargs:
            s.append("(")
            s.append(nl_indent)
            for k, v in self.kwargs.items():
                _v = str(v) \
                    if isinstance(v, (Call, Call._Partial)) \
                    else repr(v)
                head = k+"="
                s.append(head)
                _nl_indent = nl_indent + " "*len(head)
                s.append(nl_indent.join(_v.split("\n")))
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
        if self._partial:
            s.append(".partial()")
        return "".join(s)

    def partial(self):
        return Call._new(self.reference, args=self.args, kwargs=self.kwargs,
                         partial=True)

    def plain(self):
        return Call._new(self.reference, args=self.args, kwargs=self.kwargs,
                         partial=False)

    @staticmethod
    def _v1(evaluable, args, kwargs, partial):
        return Call._new(evaluable, args=args, kwargs=kwargs, partial=partial)

    @staticmethod
    def _v0(evaluable, args, kwargs):  # Legacy code for unpickling old calls
        return Call._new(evaluable, args=args, kwargs=kwargs, partial=False)

    class _Partial(Evaluable):  # legacy code for unpickling old partial calls
        @staticmethod
        def _v0(call):
            return call.partial()


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
    ... _return(c)
    ... ''')
    >>> env.run("x")
    3

    >>> env["y"] = Exec('''
    ... #@export d
    ... d = someValues["a"] + someValues["b"]
    ... e = d + someValues["a"] + someValues["b"]
    ... ''')
    >>> env.run("y", gen_refs=True) == {"d": 3}
    True
    >>> env.run("d")
    3
    """
    def __init__(self, code):
        # from textwrap import _leading_whitespace_re
        _leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])',
                                            re.MULTILINE)
        code = code.split("\n")
        _ind = min(len(x.group(1))
                   for line in code
                   for x in (_leading_whitespace_re.match(line),)
                   if x is not None)
        code = "\n".join(line[_ind:] for line in code).strip()
        self.code = code

    def __reduce__(self):
        return Exec._v0, (self.code,)

    def eval(self, env=None, *, leak_in=None, leak_out=None, gen_refs=False):
        _result = _no_value
        def _return(result):
            raise Exec._ReturnException(result)
        # Build the python environment
        leak_in = dict(leak_in) if leak_in is not None else {}
        leak_in.update(_env=env, _return=_return)
        locs = Exec._Locals(env, leak_in)
        # Execute, handle return and leak_out updating
        try:
            code = compile(self.code, "<Exec@{:X}>".format(id(self)),
                           mode="exec")
            exec(code, locs)
        except Exec._ReturnException as ret_exc:
            _result = ret_exc.args[0]
        finally:
            if leak_out is not None:
                leak_out.update(locs)
        if _result is not _no_value:
            return _result
        # Handle the return and the generation of new references if needed
        to_export = self._to_export(env=env, leak_in=leak_in, locals=locs)
        locs = {k:v for k, v in locs.items() if k in to_export}
        if gen_refs:
            # We prefer not be in an exec running an Exec
            assert env[env.current] == self, ("Cannot run a non root Exec "
                                              "with gen_refs option")
            for n in locs:
                ref = Ref("{}@{}".format(env.current, n))
                # We most likely simply want to override it.
                # current_item = env.get(n, _no_value)
                # assert current_item is _no_value or ref == current_item, \
                #     ("Variable {!r} was already defined as {!r}." 
                #      .format(n, current_item))
                env[n] = ref  # Set and erase the potential cache
        if env is not None:
            current = env.current
            for n, v in locs.items():
                env.requires[n].add(current)
                env.required_by[n].add(current)
                env.cache[n] = v
            
            env.requires[current].update(locs)
            env.required_by[current].update(locs)
        return locs

    def _to_export(self, *, env, leak_in, locals):
        # If there are @export comments, use them, otherwise, all non
        # _-beginning variables are to_export
        to_export = set(sum((line.split()[1:]
                            for line in self.code.split("\n")
                            if line.startswith("#@export ")), []))
        if not to_export:
            to_export = set(k for k in locals if not k.startswith("_"))
        return to_export

    class _ReturnException(Exception):
        pass

    class _Locals(dict):
        def __init__(self, env, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.env = env

        def __getitem__(self, key):
            value = super().get(key, _no_value)
            if value is not _no_value:
                return value
            if self.env.can_run(key):
                try:
                    return self.env.run(key)
                except Exception as e:
                    raise Exception("An exception was thrown while "
                                    "computing {!r}".format(key))
            else:
                raise KeyError(key)

        def __setitem__(self, key, value):
            if key in self.env.frozen:
                return
            super().__setitem__(key, value)

    def __str__(self):
        body = repr(self.code)
        indent = " "*3
        nl_indent = "\n" + indent
        if "\\n" in body:
            quote = body[0]
            body = body[1:-1].replace("\\n", nl_indent)
            body = ("{quote}{quote}{quote}\n{indent}"
                    "{}\n"
                    "{quote}{quote}{quote}"
                    .format(body, quote=quote, indent=indent))
        return "Exec({})".format(body)

    def __repr__(self):
        return "Exec({!r})".format(self.code)

    @staticmethod
    def _v0(code):
        return Exec(code)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running_stack = []
        self.clean()

    def can_run(self, name):
        return name in self or name in self.cache

    def __reduce__(self):
        return Environment._v0, (dict(self),)

    def __getitem__(self, key):
        """
        >>> env = Environment(training_keys="a", testing_keys="b")
        >>> env["training_keys"]
        'a'
        >>> env[["training_keys", "testing_keys"]]
        ['a', 'b']
        """
        if isinstance(key, list):
            return [self[k] for k in key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """
        >>> env = Environment()
        >>> env["a"] = "b"
        >>> env.update(b="c")
        >>> env.run(["a", "b"])
        ['b', 'c']
        """
        self._clean(key)
        return super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        to_remove = {}  # empty dict on which we simulate the update
        to_remove.update(*args, **kwargs)
        self._clean(*to_remove)
        super().update(*args, **kwargs)
        return self

    @property
    def current(self):
        return self.running_stack[-1]

    def copy(self, with_cache=True, with_frozen_values=True):
        copy = Environment(self)
        if with_cache:
            copy.cache.update(self.cache)
            copy.required_by.update(self.required_by)
            copy.requires.update(self.requires)
            if with_frozen_values:
                copy.frozen.update(self.frozen)
        return copy

    def freeze(self, *names, **kwargs):
        """Freeze an item's value (when assigned into Exec code at the moment)
        """
        self._clean(*names, *kwargs)
        self.cache.update(kwargs)
        self.frozen.update(names, kwargs)
        return self

    def feed(self, feed_dict, feed_cache=True):
        """
        >>> env = Environment(a=Call(print).with_args("Compute a"),
        ...                   b=Call(print).with_args("Compute b", "@a"))
        >>> env.run("a")
        Compute a
        >>> env.copy().feed({"a":", great!"}).run("b")
        Compute b , great!
        >>> env.run("b")
        Compute b None
        """
        if feed_cache:
            self._clean(*feed_dict)
            self.cache.update(feed_dict)
        else:
            self.update(feed_dict)
        return self

    def run(self, name, **kwargs):
        if isinstance(name, list):
            return [self.run(n, **kwargs) for n in name]
        for running in self.running_stack:
            # For computing running, we need "name"
            self.requires[running].add(name)
            self.required_by[name].add(running)
        value = self.cache.get(name, _no_value)
        if value is _no_value:
            # Not in the cache, evaluate if Evaluable
            value = self[name]
            if isinstance(value, Evaluable):
                self.running_stack.append(name)
                try:
                    value = value.eval(self, **kwargs)
                finally:
                    self.running_stack.pop()
            self.cache[name] = value
        return value

    def force_run(self, name, **kwargs):
        if isinstance(name, list):
            return [self.force_run(n, **kwargs) for n in name]
        for running in self.running_stack:
            # For computing running, we need "name"
            self.requires[running].add(name)
            self.required_by[name].add(running)
        # Evaluate whatever happens
        value = self[name]
        if isinstance(value, Evaluable):
            self.running_stack.append(name)
            try:
                value = value.eval(self, **kwargs)
            finally:
                self.running_stack.pop()
        self.cache[name] = value
        return value

    @property
    def fused(self):
        cache_without_self = {k: v
                              for k, v in self.cache.items()
                              if k != "_env" or "_env" in self
                             }
        return Environment({**self, **cache_without_self})

    def __delitem__(self, key):
        """
        >>> env = Environment(a=3, b=4)
        >>> env.run(["a", "b"])
        [3, 4]
        >>> del env["a"]
        >>> env.run("a")
        Traceback (most recent call last):
            ...
        KeyError: 'a'
        >>> env.pop("b")
        4
        >>> env.run("b")
        Traceback (most recent call last):
            ...
        KeyError: 'b'
        """
        self.cache.pop(key, None)
        self.frozen.discard(key)
        return super().__delitem__(key)
    
    def pop(self, key, *args):
        self.cache.pop(key, None)
        self.frozen.discard(key)
        return super().pop(key, *args)

    def clean(self):
        if self.get("_env", _no_value) is _no_value:
            self.cache = dict(_env=self)
        else:
            self.cache = {}
        self.required_by = defaultdict(set)
        self.requires = defaultdict(set)
        self.frozen = set()

    def _clean(self, *names):
        """Cleans the dependency graph
        >>> env = Environment(
        ...     a=Call(print).with_args("a", Ref("b")),
        ...     b=Call(print).with_args("b", Ref("c")),
        ...     c=Call(print).with_args("c", "Finally!")
        ... )
        >>> env.run("a")
        c Finally!
        b None
        a None
        >>> env["b"] = None
        >>> list(env.cache.keys())
        ['_env', 'c']
        >>> env.run("a")
        a None
        >>> env["c"] = None
        >>> set(env.cache.keys()) == set(['_env', 'a', 'b'])
        True
        """
        to_clean = list(names)
        while to_clean:
            # what basically has to happen is that we want to remove the cache
            # and the requirements (requires.pop) and remove the entries that 
            # requires this one
            name = to_clean.pop()
            self.cache.pop(name, None)
            self.frozen.discard(name)
            # requirements of "name" will be recomputed on next evaluation
            requirements = self.requires.pop(name, ())
            # the requirements should not clean "name" anymore
            for req in requirements:
                self.required_by[req].discard(name)
            # clean the ones that require "name"
            to_refresh = self.required_by.pop(name, ())
            to_clean.extend(to_refresh)

    def __str__(self):
        indentation = " "*3
        nl_indent = "\n"+indentation
        s = ["Environment("]
        if self:
            for k, v in self.items():
                s.append(nl_indent)
                head = "{}=".format(k)
                s.append(head)
                _nl_indent = nl_indent + " "*len(head)
                _v = str(v) \
                    if isinstance(v, (Call, Call._Partial, Environment,
                                      Exec)) \
                    else repr(v)
                s.append(nl_indent.join(_v.split("\n")))
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
            if "leak_in" in flags:
                assert name == "/", \
                    ("Leaking-in is only valid with '/' as name. Indeed, you "
                     "will not want your environment to contain all of the "
                     "IPython global vars")
                # Filter the IPython's global ns for the items the environment
                # will compute.
                leak_in = {k: v
                           for k, v in _ip.user_global_ns.items()
                           if k not in env}
            else:
                leak_in = None
            leak_out = _ip.user_global_ns
            if name == "/":
                res = Exec(cell).eval(env, leak_in=leak_in, leak_out=leak_out)
            else:
                env[name] = Exec(cell)
                res = env.run(name, leak_in=leak_in, leak_out=leak_out,
                              gen_refs="no_ref" not in flags)
            if "show" in flags:
                return res


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
