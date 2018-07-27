from mlworkflow import DataCollection, LivePanels, find_files
from mlworkflow.utils import kwonly_from_ctx, DictObject
from ipywidgets import (Select, VBox, HBox, Textarea, Button, Text, Output,
                        Label)
from IPython import display
import os
import re


__all__ = ["Dashboard",
           # Widgets
           "comment", "list_files", "recording", "tags", "vbox",
           "list_properties", "filename", "execute",
]


def Dashboard(widgets):
    inst_grid = []
    _on_init = []
    def on_init(callback):
        _on_init.append(callback)
        return callback
    ctx = {"on_init": on_init}
    for row in widgets:
        inst_row = []
        inst_grid.append(inst_row)
        for widget in row:
            if widget is not Ellipsis:
                inst_row.append(widget(ctx=[locals(), ctx]))
            else:
                inst_row[-1]
    display.display(VBox([HBox(inst_row) for inst_row in inst_grid]))
    for callback in _on_init:
        callback()


def list_files(path="**.dcp", key=None, reverse=False):
    @kwonly_from_ctx
    def list_files(*, on_init, ctx):
        if isinstance(path, str):
            files = find_files(path)
        else:
            files = path
        files.sort(key=key, reverse=reverse)
        selector = Select(options=files, continuous_update=False)
        on_init(lambda: update_selected_file({"new":files[0]}))
        # Allow other widgets to use on_file_selection to add callbacks
        _on_file_selection = []
        def on_file_selection(callback):
            _on_file_selection.append(callback)
            return callback
        ctx["file_selector"] = selector
        ctx["on_file_selection"] = on_file_selection
        # Register the loading of the selected file
        def update_selected_file(change):
            filename = change["new"]
            ctx["filename"] = filename
            ctx["data"] = DataCollection.load_file(filename)
            ctx["metadata"] = DataCollection.get_metadata(filename)
            for callback in _on_file_selection:
                callback()
        selector.observe(update_selected_file, names="value")
        return selector
    return list_files


def vbox(*decls):
    def vbox(*, ctx):
        return VBox([decl(ctx=ctx) for decl in decls])
    return vbox


def comment():
    @kwonly_from_ctx
    def comment(*, on_file_selection, ctx):
        content = Textarea(placeholder="Any comment?")
        submit = Button(description="Save comment")
        @on_file_selection
        def update_content():
            content.value = ctx["metadata"].get("comment", "")
        @submit.on_click
        def on_click(_):
            DataCollection.add_metadata(ctx["filename"], {"comment":content.value})
        return VBox([content, submit])
    return comment


def tags():
    @kwonly_from_ctx
    def tags(*, on_file_selection, ctx):
        tags = Text()
        button = Button(description="Save Tags")
        @on_file_selection
        def update_tags():
            tags.value = " ".join(ctx["metadata"].get("tags", []))
        @button.on_click
        def on_click(_):
            DataCollection.add_metadata(ctx["filename"], {"tags":tags.value.split()})
        return HBox([tags, button])
    return tags


def recording(fields=["head", "body"]):
    @kwonly_from_ctx
    def recording(*, on_file_selection, ctx):
        slider = Output()
        _panels = {}
        for field in fields:
            _panels[field] = Output()
        panels = LivePanels(fields, panels=_panels, slider=slider)
        @on_file_selection
        def update_content():
            panels.show_recording(ctx["data"][:,"recording"])
        return VBox((slider,)+tuple(_panels.values()))
    return recording


def filename(tag="h3", link_parent=False):
    def parent_path(child_data):
        parent = child_data[-1,"_parent":None]
        if parent is not None:
            parent = os.path.join(os.path.dirname(child_data.filename), parent)
            parent = os.path.normpath(parent)
        return parent

    @kwonly_from_ctx
    def filename(*, file_selector, on_file_selection, ctx):
        widget = html = Output()
        if link_parent:
            parent_label = Label()
            parent_button = Button(description="show")
            @parent_button.on_click
            def show_parent(_):
                file_selector.value = parent_path(ctx["data"])
            widget = VBox([html, HBox([parent_label, parent_button])])
        @on_file_selection
        def update():
            if link_parent:
                parent = parent_path(ctx["data"])
                if parent is not None:
                    parent_label.value = "Parent: "+parent
                    parent_button.disabled = False
                else:
                    parent_label.value = "No parent"
                    parent_button.disabled = True
            with html:
                display.clear_output()
                display.display(display.HTML("<{tag}>{filename}</{tag}>".format(tag=tag, filename=ctx["filename"])))
        return widget
    return filename


def list_properties():
    @kwonly_from_ctx
    def list_properties(*, on_file_selection, ctx):
        output = Output()
        @on_file_selection
        def update_html():
            data = ctx["data"]
            metadata = ctx["metadata"]
            dkeys = set()
            dkeys.update(*data)
            with output:
                display.clear_output()
                display.display(display.HTML("<b>Data</b>: {}<br /><b>Metadata</b>: {}".format(
                    " ".join(sorted(dkeys)), " ".join(sorted(metadata))
                )))
        return output
    return list_properties


def execute(globals, field="comment"):
    @kwonly_from_ctx
    def exec_code(*, ctx, on_file_selection):
        output = Output()
        @on_file_selection
        def update():
            text = ctx["metadata"].get(field, "")
            code_sections = re.findall(r"```(.*?)```", text,
                                       re.MULTILINE|re.DOTALL)
            with output:
                display.clear_output()
                env = {**globals,
                       "data": ctx["data"]}
                for code in code_sections:
                    exec(code, env)
        return output
    return exec_code
