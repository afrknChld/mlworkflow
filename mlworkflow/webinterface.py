import argparse
import bokeh
from bokeh.server.server import Server
from bokeh.models.widgets import Div
from mlworkflow.file_handling import find_files
from types import ModuleType
from bokeh.io.doc import set_curdoc
import sys
import os

from bokeh.io import curdoc


def dispatcher(filenames):
    filenames = filenames.split(",")
    def dispatch(doc):
        files = find_files(filenames)
        html = []
        sep = ""
        for i, file in enumerate(files):
            html.append(f'''{sep}<a style="color:purple;text-decoration:none;" href="?app={file}">{file}</a>''')
            next_ = files[i+1:]
            sep = "<br/>" if next_ and os.path.dirname(next_[0]) != os.path.dirname(file) else " | "
        doc.add_root(Div(text="".join(html),
                         style=dict(color="black",
                                    overflowY="scroll",maxHeight="10vh"
                                    )
                         ))

        _args = doc.session_context.request.arguments
        args = {k: v[0].decode("utf-8") for k, v in _args.items()}
        app = args.get("app", None)
        if app is not None:
            app = os.path.normpath(app)
        if app is not None and not (app.startswith("..") or os.path.isabs(app)):
            with open(app, "r") as file:
                source = file.read()
            set_curdoc(doc)
            module_name = app[:-3].replace("/", ".")
            module = ModuleType(module_name)
            sys.modules[module_name] = module
            module.__file__ = app
            exec(source, module.__dict__)
            del sys.modules[module_name]
    return dispatch

if __name__ == '__main__':
    # From bokeh/command/subcommands/serve.py
    from bokeh.resources import DEFAULT_SERVER_PORT
    args = (
        ('--port', dict(
            metavar = 'PORT',
            type    = int,
            help    = "Port to listen on",
            default = DEFAULT_SERVER_PORT
        )),

        ('--address', dict(
            metavar = 'ADDRESS',
            type    = str,
            help    = "Address to listen on",
            default = None,
        )),

        ('--allow-websocket-origin', dict(
            metavar='HOST[:PORT]',
            action='append',
            type=str,
            help="Public hostnames which may connect to the Bokeh websocket",
        )),

        ('--apps', dict(
            default="*/app_*.py",
            type=str,
            help="Apps to load",
        ))
    )
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(arg[0], **arg[1])
    server_args = parser.parse_args()

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.
    server = Server({'/': dispatcher(server_args.apps)}, debug=True, **vars(server_args))
    server.start()

    print(f'Opening Tornado app with embedded Bokeh application on http://localhost:{server.port}/')
    server.io_loop.start()
