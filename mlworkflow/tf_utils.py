from mlworkflow.utils import DictObject
from contextlib import contextmanager
from functools import wraps


def _callable_with_defaults(session, fetches, feed_list, *, silent_fetches=[], feed_default={},
                            accept_options=False, doc=None):
    default_keys = list(feed_default.keys())
    default_values = tuple(feed_default.values())
    if isinstance(fetches, (tuple, list)):
        fetches_range = slice(len(fetches))
        fetches = [*fetches, *silent_fetches]
    else:
        fetches_range = 0
        fetches = [fetches, *silent_fetches]
    tf_call = session.make_callable(fetches, [*default_keys, *default_values],
                                    accept_options=accept_options)
    f = lambda *args, **kwargs: tf_call(*default_values, *args, **kwargs)[fetches_range]
    if doc is not None:
        f.__doc__ = doc
    return f


class TFModel:
    def __init__(self, initializer=None, session=None, graph=None,
                 initialize_global_variables=True, device="/gpu:0",
                 allow_soft_placement=True, log_device_placement=False):
        import tensorflow as tf
        self._dedicated_session = session is None
        if session is None:
            if graph is None:
                graph = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                    log_device_placement=log_device_placement)
            session = tf.Session(graph=graph, config=config)
        else:
            assert graph is None or graph is session._graph, ("Cannot use a "
                "graph different than that of the provided session")
            graph = session._graph
        self.graph = graph
        self.session = session
        self.initializer = initializer
        self.exports = None
        self.device = device

        if self.exports is None and self.initializer is not None:
            with self.graph.as_default(), self.session.as_default():
                with tf.device(self.device):
                    self.exports = DictObject(self.initializer())
                if initialize_global_variables:
                    self.session.run(tf.global_variables_initializer())
        self._train = self._predict = self._evaluate = None
        self._evaluation_losses = None

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.__getitem__(k) for k in key]
        if isinstance(key, str):
            key = self.exports[key]
        return self.session.run(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            for k, v in zip(key, value):
                self.__setitem__(k, v)
            return
        if isinstance(key, str):
            key = self.exports[key]
        key.load(value, session=self.session)

    def train(self, inputs, targets):
        if self._train is None:
            self._train = self.session.make_callable([self.exports.loss, self.exports.step],
                [self.exports.training, self.exports.inputs, self.exports.targets])
        return self._train(True, inputs, targets)[0]

    def predict(self, inputs):
        if self._predict is None:
            self._predict = self.session.make_callable([self.exports.outputs],
                [self.exports.training, self.exports.inputs])
        return self._predict(False, inputs)[0]

    def evaluate(self, inputs, targets, losses=None):
        if self._evaluation_losses != losses or self._evaluate is None:
            self._evaluation_losses = losses
            if losses is None:
                losses = self.exports.loss
            elif isinstance(loss, str):
                losses = self.exports[losses]
            elif isinstance(loss, (tuple, list)):
                losses = [self.exports[loss] if isinstance(loss, str) else loss
                            for loss in losses]
            else:
                losses = losses
            self._evaluate = self.session.make_callable(losses,
                [self.exports.training, self.exports.inputs, self.exports.targets])
        return self._evaluate(False, inputs, targets)

    def run(self, fetches, feed_dict):
        fetches = [self.exports[fetch] if isinstance(fetch, str) else fetch
                   for fetch in fetches]
        feed_dict = {self.exports[key]:v if isinstance(key, str) else key
                     for key, v in feed_dict.items()}
        return self.session.run(fetches, feed_dict=feed_dict)

    @staticmethod
    def from_function(initializer=None, **kwargs):
        def from_function(initializer):
            return TFModel(initializer=initializer, **kwargs)
        if initializer is None:
            return from_function
        return from_function(initializer)

    def __del__(self):
        if self._dedicated_session:
            self.session.close()

    def get_variables(self, variables=""):
        prefix = lst = variables
        variables = self.graph.get_collection("variables")
        if isinstance(prefix, str):
            variables = [var for var in variables
                         if var.name.startswith(prefix)]
        elif isinstance(lst, (tuple, list, set)):
            variables = [var for var in variables
                         if var in lst]
        names = [var.name for var in variables]
        return dict(zip(names, self.session.run(variables)))

    def set_variables(self, feed_dict):
        variables = [var for var in self.graph.get_collection("variables")
                     if var.name in feed_dict]
        for var in variables:
            var.load(feed_dict[var.name], session=self.session)


def tf_name_scope(f=None, *, name=None):
    def tf_name_scope(f):
        nonlocal name
        if name is None:
            name = f.__name__
        @wraps(f)
        def name_scope_wrapper(*args, **kwargs):
            import tensorflow as tf
            with tf.name_scope(name):
                return f(*args, **kwargs)
        return name_scope_wrapper
    if f is None:
        return tf_name_scope
    return tf_name_scope(f)


def tf_name_scope(f=None, *, name=None):
    def tf_name_scope(f):
        nonlocal name
        if name is None:
            name = f.__name__
        @wraps(f)
        def name_scope_wrapper(*args, **kwargs):
            import tensorflow as tf
            with tf.name_scope(name):
                return f(*args, **kwargs)
        return name_scope_wrapper
    if f is None:
        return tf_name_scope
    return tf_name_scope(f)
