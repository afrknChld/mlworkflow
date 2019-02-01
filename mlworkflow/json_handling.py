from importlib import import_module
import json
import re


_comment_remover = re.compile(r'//[^\n]*|/\*.*?\*/', re.RegexFlag.MULTILINE|re.RegexFlag.DOTALL)
_comma_remover = re.compile(r',\s*([\}\]])')


def _replacer(match):
    return match.group(1)


def remove_comments(jsonc_string):
    jsonc_string = re.sub(_comment_remover, '', jsonc_string)
    jsonc_string = re.sub(_comma_remover, _replacer, jsonc_string)
    return jsonc_string


def loads_jsonc(s):
    s = remove_comments(s)
    s = json.loads(s)
    return s


def load_jsonc(fp):
    return loads_jsonc(fp.read())


def _copy(json, *, root, utils):
    qualname = json["_copy"]
    elem = root
    for n in qualname.split("."):
        if isinstance(elem, dict):
            elem = elem[n]
        elif isinstance(elem, list):
            elem = elem[int(n)]
        else:
            raise NotImplementedError()
    return elem
_utils = dict(_copy=_copy)


def preprocess(json, root=None, utils=_utils):
    """Should be called first on JSON-loaded dict (contains only dict and lists)
    This is simply meant to add some syntactic sugar.
    """
    if root is None:
        root = json
    if isinstance(json, dict):
        if len(json) == 1:
            k = next(iter(json))
            if k in utils:
                return utils[k](json, root=root, utils=utils)
        parsed = {}
        for k, v in json.items():
            parsed[k] = preprocess(v, root=root, utils=utils)
        return parsed
    elif isinstance(json, list):
        return [preprocess(l, root=root, utils=utils) for l in json]
    return json


def update_dict(dic, update):
    if isinstance(update, dict):
        update = update.items()
    for qualname, value_to_set in update:
        names = qualname.split(".")
        d = dic
        for n in names[:-1]:
            if isinstance(d, dict):
                d = d[n]
            elif isinstance(d, list):
                d = d[int(n)]
            else:
                raise NotImplementedError()

        n = names[-1]
        if isinstance(d, dict):
            d[n] = value_to_set
        elif isinstance(d, list):
            d[int(n)] = value_to_set


def eval_json(json, env):
    """Should be called 2nd, after preprocessing. Simply meant to allow more complicated
    structures (e.g. creating of dict with int keys) from JSON"""
    if isinstance(json, dict):
        parsed = {}
        for k, v in json.items():
            parsed[k] = eval_json(v, env)
        call = parsed.pop("_call", None)
        if call is not None:
            if call.startswith("!"):
                parsed["_call"] = call[1:]
                return parsed
            if isinstance(call, str):
                call = env[call]
            args = parsed.pop("_args", [])
            parsed = call(*args, **parsed)
        return parsed
    elif isinstance(json, list):
        return [eval_json(l, env) for l in json]
    return json


def _resolve_calls(json, env):
    """Instantiates a Call from a JSON _call
    
    not reducible to eval_json we just go get the references of the functions
    in the environment
    """
    if isinstance(json, dict):
        parsed = {}
        for k, v in json.items():
            parsed[k] = eval_json(v, env)
        call = parsed.pop("_call", None)
        if call is not None:
            if call.startswith("!"):
                parsed["_call"] = call[1:]
                return parsed
            if isinstance(call, str):
                call = env[call]
            args = parsed.pop("_args", [])
            parsed = Call(call, args=args, kwargs=parsed)
        return parsed
    elif isinstance(json, list):
        return [_resolve_calls(l, env) for l in json]
    return json


class Call(dict):
    def __init__(self, fun, module=None, *, args=[], kwargs={}, partial=False):
        super().__init__()
        self["_call"] = "Call"
        if module is not None:
            self["fun"] = fun
            self["module"] = module
        elif callable(fun):
            self["fun"] = fun.__qualname__
            self["module"] = fun.__module__
        else:
            raise NotImplementedError()
        self["args"] = args
        self["kwargs"] = kwargs
        self["partial"] = partial

    def on(self, fun, module=None):
        return Call(fun, module,
                    args=self["args"], kwargs=self["kwargs"],
                    partial=self["partial"])

    def with_args(self, *args):
        new_args = sum(((arg,) if arg is not Ellipsis else self["args"]
                        for arg in args
                        ), ())
        return Call(self["fun"], self["module"],
                    args=new_args, kwargs=self["kwargs"],
                    partial=self["partial"])

    def __call__(self, **kwargs):
        return Call(self["fun"], self["module"],
                    args=self["args"], kwargs={**self["kwargs"], **kwargs},
                    partial=self["partial"])

    def partial(self):
        return Call(self["fun"], self["module"],
                    args=self["args"], kwargs=self["kwargs"],
                    partial=True)

    def plain(self):
        return Call(self["fun"], self["module"],
                    args=self["args"], kwargs=self["kwargs"],
                    partial=False)

    @staticmethod
    def resolve(json, env):
        """Go from _call to Call"""
        return _resolve_calls(json, env)

    @staticmethod
    def instantiate(json, env={}):
        """Go from Call dict to Call"""
        return eval_json(json, dict(**env, Call=Call))

    def eval(self, env={}):
        """Evaluate a Call"""
        return eval_json(self, dict(**env, Call=Call._eval_call))

    @staticmethod
    def _eval_call(fun, module, args, kwargs, partial):
        callee = import_module(module)
        for f in fun.split("."):
            callee = getattr(callee, f)
        if partial:
            lambda *args, **kwargs: callee(*args, **kwargs)
        return callee(*args, **kwargs)

    # def __repr__(self):
    #     kwargs = args = partial = ""
    #     if self["args"]:
    #         args = ".with_args({})".format(", ".join(repr(arg)
    #                                                  for arg in self["args"]))
    #     if self["kwargs"]:
    #         kwargs = "({})".format(", ".join("{}={!r}".format(k, v)
    #                                          for k, v in self["kwargs"].items()))
    #     if self["partial"]:
    #         partial = ".partial()"
    #     return "Call({}.{}){}{}{}".format(self["module"], self["fun"],
    #                                       args, kwargs, partial)

    # def __str__(self):
    #     s = ["Call({}.{})".format(self["module"], self["fun"])]
    #     indentation = " "*3
    #     nl_indent = "\n{}".format(indentation)
    #     if self["kwargs"]:
    #         s.append("(")
    #         s.append(nl_indent)
    #         for k, v in self["kwargs"].items():
    #             _v = str(v) \
    #                 if isinstance(v, Call) \
    #                 else repr(v)
    #             head = k+"="
    #             s.append(head)
    #             _nl_indent = nl_indent + " "*len(head)
    #             s.append(nl_indent.join(_v.split("\n")))
    #             s.append(",\n" + indentation)
    #         s[-1] = s[-1][1:-1]  # remove comma and space
    #         s.append(")")
    #     if self["args"]:
    #         s.append(".with_args(")
    #         s.append(nl_indent)
    #         for arg in self["args"]:
    #             _v = str(arg) if isinstance(arg, Call) else repr(arg)
    #             s.append(nl_indent.join(_v.split("\n")))
    #             s.append(",\n" + indentation)
    #         s[-1] = s[-1][1:-1]  # remove comma and space
    #         s.append(")")
    #     if self["partial"]:
    #         s.append(".partial()")
    #     return "".join(s)
