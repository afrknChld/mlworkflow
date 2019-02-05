from bokeh.models import InputWidget
from bokeh.core.properties import String, Instance
from bokeh.models.callbacks import Callback


class TextArea(InputWidget):
    ''' Textarea widget.

    '''

    __implementation__ = "textarea.ts"
    __javascript__ = []
    __css__ = []

    value = String(default="", help="""
    Initial or entered text value.
    """)

    callback = Instance(Callback, help="""
    A callback to run in the browser whenever the user unfocuses the TextInput
    widget by hitting Enter or clicking outside of the text box area.
    """)

    placeholder = String(default="", help="""
    Placeholder for empty textarea field
    """)