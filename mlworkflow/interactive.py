from IPython.core.displaypub import DisplayPublisher
from ipywidgets import Output, IntSlider, interact
from traitlets import List, Any
from IPython import get_ipython, display

import functools
import sys


class MirrorDisplayPublisher(DisplayPublisher):
    """A DisplayPublisher that is meant to replace get_ipython.display_pub. It
    will write the publications to a list as well as appending them to its
    outputs list.
    """
    outputs = List()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_publisher = get_ipython().display_pub

    def publish(self, data, metadata=None, source=None):
        self.outputs.append({"output_type": "display_data", "data": data,
                             "metadata": metadata, "source": source})
        return self.root_publisher.publish(data, metadata, source)

    def clear_output(self, wait=False):
        super().clear_output(wait)
        del self.outputs[:]  # Empty the list
        return self.root_publisher.clear_output(wait)


class MirrorWritable:
    r"""Meant to replace sys.stdout or sys.stderr for sending writes to the
    outputs list as well as mirroring them.

    >>> outputs = []
    >>> mirror = MirrorWritable(outputs, "stdout")
    >>> mirror.write("hello\n")  # still returns the number of bytes written
    hello
    6
    >>> outputs == [{'output_type': 'stream', 'name': 'stdout',
    ...              'text': 'hello\n'}]
    True
    """
    def __init__(self, outputs, stream_name, mirror=True):
        self.outputs = outputs
        self.stream_name = stream_name
        self.stream = getattr(sys, stream_name)

    def write(self, text):
        self.outputs.append({"output_type": "stream", "name": self.stream_name,
                             "text": text})
        return self.stream.write(text)

    def __getattr__(self, key):
        return getattr(self.stream, key)


class forward_output:
    """A context manager that modifies streams and publisher, and passes the
    hand to an optional ipywidget Output.

    This version simply forwards to the messages to the ones ruling at its
    initialization.
    """
    def __init__(self, output):
        self.output = output
        self.ip = get_ipython()
        self._init()

    def _init(self):
        self.display_pub = self.ip.display_pub
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.outputs = None

    def __enter__(self):
        self._display_pub = self.ip.display_pub
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        self.ip.display_pub = self.display_pub
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if self.output is not None:
            self.output.__enter__()
        return self.outputs

    def __exit__(self, etype, evalue, tb):
        if self.output is not None:
            self.output.__exit__(etype, evalue, tb)
        self.ip.display_pub = self._display_pub
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class record_output(forward_output):
    """A context manager that modifies streams and publisher, and passes the
    hand to a potential ipywidget Output.

    This version redirects streams to mirrors in order to record

        with record_output() as recorded:
            print("hello!")
        # prints 'hello!'
        recorded
        # contains [{'output_type': 'stream', 'name': 'stdout',
        #            'text': 'hello\n'}]
    """
    def __init__(self, output=None):
        super().__init__(output)

    def _init(self):
        self.display_pub = MirrorDisplayPublisher()
        self.outputs = self.display_pub.outputs
        self.stdout = MirrorWritable(self.outputs, "stdout")
        self.stderr = MirrorWritable(self.outputs, "stderr")


def replay_output(outputs):
    ip = get_ipython()
    for output in outputs:
        if isinstance(output, dict):
            if output["output_type"] == "stream":
                getattr(sys, output["name"]).write(output["text"])
            elif output["output_type"] == "display_data":
                ip.display_pub.publish(output["data"],
                                       output.get("metadata", None),
                                       output.get("source", None))
        if isinstance(output, tuple):   # TODO: Remove, breaks backward
                                        # compatibility
            if output[0] == "write":
                getattr(sys, output[1]).write(output[2])
            elif output[0] == "publish":
                ip.display_pub.publish(output[1], output[2], output[3])


try:
    import tqdm as tqdm_module
except ImportError:
    tqdm_module = None


class LivePanels:
    """Panels integrating with IPython for interactive outputs
    from IPython import display

    # Records the body of the loop (tqdm happens in "head")
    panels = LivePanels(record=["body"])
    for i in panels.tqdm(range(10)):
        display.clear_output(wait=True)
        print(i)

    # A slider is proposed for visualizing the output of the different
    # iteration outputs

    You may add panels and output into them by doing

    panels = LivePanels(["head", "mine", "body"])
    with panels.mine:
        print("This will show up between tqdm and the iteration output")

    Or do

    panels["mine"] = "Overwritable line"
    """
    def __init__(self, panel_names=["head", "body"], *, record=[], panels=None,
                 slider=None):
        if slider is None:
            slider = Output()
            display.display(slider)
        if panels is None:
            panels = {}
            for panel_name in panel_names:
                output = Output()
                panels[panel_name] = output
                display.display(output)
        self._slider = slider
        self.panels = panels

        self._recorders = {n: record_output(self.panels[n])
                           for n in record} if record else None
        # Every output must be backed, otherwise, a non-recorder inside a
        # recorder will be recorded.
        self._proxies = {n: (self._recorders[n]
                             if n in self._recorders
                             else forward_output(output))
                         for n, output in self.panels.items()} \
            if record else None

        self.recording = [] if record else None

        self._head_name = "head"
        self._body_name = "body"

        self._head = getattr(self, self._head_name)
        self._body = getattr(self, self._body_name)
        self._max_length_to_overwrite = {}

    def __getattr__(self, panel_name):
        if self._proxies is not None:
            ctx_manager = self._proxies.get(panel_name, None)
            if ctx_manager is not None:
                return ctx_manager
        return self.panels[panel_name]

    def __setitem__(self, panel_name, value):
        with getattr(self, panel_name):
            maxlen = self._max_length_to_overwrite.get(panel_name, 0)
            value = str(value)
            curlen = len(value)
            if curlen > maxlen:
                self._max_length_to_overwrite[panel_name] = curlen
            sys.stdout.write("\r{:<{}}".format(value, maxlen))

    def show_recording(self, recording):
        def show_output(iteration):
            outputs = recording[iteration]
            for name, output in outputs.items():
                with self.panels[name]:
                    display.clear_output(wait=True)
                    replay_output(output)
        n_iterations = len(recording)
        with self._slider:
            display.clear_output(wait=True)
            if n_iterations > 0:
                slider = IntSlider(min=0, max=n_iterations-1,
                                   value=n_iterations-1,
                                   continuous_update=False)
                interact(show_output, iteration=slider)

    @property
    def current_record(self):
        record = {n: [*rec.outputs]
                  for n, rec in self._recorders.items()}
        return record

    def _on_iteration_end(self):
        if self.recording is not None:
            self.recording.append(self.current_record)

    def _on_loop_end(self):
        if self.recording is not None:
            self.show_recording(self.recording)

    def iterate(self, iterator, caption, *, _level=0):
        env = sys._getframe(_level+1).f_locals
        try:
            for item in iterator:
                with self._body:
                    yield item
                self["head"] = caption.format(**env)
                self._on_iteration_end()
        finally:
            self._on_loop_end()

    if tqdm_module is not None:
        @functools.wraps(tqdm_module.tqdm)
        def tqdm(self, *args, **kwargs):
            try:
                with self._head:
                    self._tqdm = iterator = tqdm_module.tqdm(*args, **kwargs)
                    for item in iterator:
                        with self._body:
                            yield item
                        self._on_iteration_end()
            finally:
                self._on_loop_end()

        @functools.wraps(tqdm_module.tqdm_notebook)
        def tqdm_notebook(self, *args, **kwargs):
            import tqdm as tqdm_module
            try:
                with self._head:
                    self._tqdm = iterator = tqdm_module.tqdm_notebook(*args,
                                                                      **kwargs)
                    for item in iterator:
                        with self._body:
                            yield item
                        self._on_iteration_end()
            finally:
                self._on_loop_end()


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
