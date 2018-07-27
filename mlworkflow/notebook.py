from IPython import get_ipython
import nbformat

import pickle
import base64
import time

from mlworkflow.utils import _exec


def run_in_cell(f=None, level=0):
    """Executes the body of a function as it is being defined
    """
    def decorator(f):
        import inspect
        import ast
        class ReturnException(Exception):
            pass
        class ReturnTransformer(ast.NodeTransformer):
            def visit_Return(self, node):
                return ast.copy_location(
                    ast.Raise(exc=ast.Call(func=ast.Name("__ReturnException",
                                                            ast.Load()),
                                            args=[node.value],
                                            keywords=[])),
                    node
                    )

        source = inspect.getsource(f)
        filename = "<string>"
        tree = ast.parse(source, filename, mode="single")
        assert isinstance(tree.body[0], ast.FunctionDef), \
            ("The run_as_cell decorator must be used with a function "
             "definition.")
        
        body_tree = tree.body[0].body
        body_tree = ast.Module(body_tree)
        body_tree = ReturnTransformer().visit(body_tree)
        body_tree = ast.fix_missing_locations(body_tree)
        body_code = compile(body_tree, filename, mode="exec")

        custom_globals = {k: v.default
                          for k, v in inspect.signature(f).parameters.items()
                          if v.default is not inspect._empty
                          }
        custom_globals["__ReturnException"] = ReturnException
        res = None

        try:
            _exec(body_code, level=level+1, custom_globals=custom_globals)
        except ReturnException as e:
            res = e.args[0]
        finally:
            f.result = res
            return f
    if f is None:
        return decorator
    else:
        level += 1
        return decorator(f)


class Notebook(nbformat.notebooknode.NotebookNode):
    def __init__(self, path, as_version=None):
        if as_version is None:
            as_version = nbformat.current_nbformat
        notebook = nbformat.read(path, as_version)
        super().__init__(notebook)

    @staticmethod
    def pickle(name, value):
        ip = get_ipython()
        value = pickle.dumps(value)
        value = base64.b64encode(value)
        value = value.decode("utf-8")
        ip.display_pub.publish({}, {"type": "mlworkflow_storage",
                                    "name": name, "value": value,
                                    "timestamp": time.time()})

    def unpickle(self, name):
        maxts = -1
        latest = None
        for cell in self.cells:
            for output in cell.outputs:
                if output["output_type"] != "display_data":
                    continue
                md = output.metadata
                if (md.get("type", None) == "mlworkflow_storage" and
                        md["name"] == name and
                        md["timestamp"] > maxts):
                    latest = md["value"]
                    maxts = md["timestamp"]
        latest = base64.b64decode(latest)
        latest = pickle.loads(latest)
        return latest

    def get_cell(self, name):
        _name = "#@name {}\n".format(name)
        cells = []
        for cell in self.cells:
            if cell["cell_type"] == "code" and _name in cell["source"]:
                cells.append(cell)
        assert len(cells) == 1, ("Found {} cells with the name "
                                 "{!r}".format(len(cells), name))
        return cells[0]

    def define(self, signature, cell_name, ret=None, level=0):
        source = self.get_cell(cell_name).source.split("\n")
        source.append("return {}".format(ret))
        source = "\n    ".join(source)
        _exec("""def {signature}:\n{body}"""
              .format(signature=signature, body=source),
              level=level+1)

    def replay(self, cell_name, level=0):
        _exec(self.get_cell(cell_name).source, level=level+1)

    def publish(self, cell_name):
        from mlworkflow.interactive import replay_output
        replay_output(self.get_cell(cell_name).outputs)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
