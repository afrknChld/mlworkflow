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


def _copy(qualname, root):
    elem = root
    for n in qualname.split("."):
        if isinstance(elem, dict):
            elem = elem[n]
        elif isinstance(elem, list):
            elem = elem[int(n)]
        else:
            raise NotImplementedError()
    return elem
def _tuple(value, root):
    return tuple(value)
def _dict(value, root):
    return dict(value)
_djson_utils = dict(_copy=_copy, _tuple=_tuple, _dict=_dict)


class DJSON:
    @staticmethod
    def from_json(json, root=None):
        if root is None:
            root = json
        if isinstance(json, dict):
            if len(json) == 1:
                k = next(iter(json))
                util = _djson_utils.get(k, None)
                if util is not None:
                    return util(DJSON.from_json(json[k], root), root)
            parsed = {}
            for k, v in json.items():
                parsed[k] = DJSON.from_json(v, root)
            return parsed
        if isinstance(json, list):
            return [DJSON.from_json(el, root) for el in json]
        return json

    @staticmethod
    def to_json(djson):
        if isinstance(djson, dict):
            if any(not isinstance(k, str) for k in djson):
                return {"_dict": [[DJSON.to_json(k), DJSON.to_json(v)]
                                  for k, v in djson.items()
                                  ]}
            else:
                transformed = {}
                for k, v in djson.items():
                    transformed[k] = DJSON.to_json(v)
                return transformed
        if isinstance(djson, list):
            return [DJSON.to_json(el) for el in djson]
        if isinstance(djson, tuple):
            return  {"_tuple": DJSON.to_json(list(djson))}
        return djson


def djsonc_loads(s, **kwargs):
    s = remove_comments(s)
    s = json.loads(s, **kwargs)
    s = DJSON.from_json(s)
    return s

def djsonc_load(fp, **kwargs):
    return djsonc_loads(fp.read(), **kwargs)


def djson_loads(s, **kwargs):
    s = json.loads(s, **kwargs)
    s = DJSON.from_json(s)
    return s

def djson_load(fp, **kwargs):
    return djson_loads(fp.read(), **kwargs)


def djson_dumps(s, separators=(',',':'), **kwargs):
    s = DJSON.to_json(s)
    s = json.dumps(s, separators=separators, **kwargs)
    return s

def djson_dump(s, fp, **kwargs):
    return fp.write(djson_dumps(s, **kwargs))


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
