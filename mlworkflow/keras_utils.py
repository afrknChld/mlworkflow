

_no_value = object()


def get_keras_structure(container, *, struct=None):
    from keras.engine.topology import Container
    if struct is None:
        struct = {}
        struct[""] = container.name
    struct[container.name] = tuple(layer.name for layer in container.layers)
    for layer in container.layers:
        if struct.get(layer.name, _no_value) is not _no_value:
            continue
        if isinstance(layer, Container):
            get_keras_structure(layer, struct=struct)
        else:
            struct[layer.name] = layer
    return struct


def get_keras_weights(container, *, weights=None):
    from keras.engine.topology import Container
    if weights is None:
        weights = {}
        weights[""] = container.name
    weights[container.name] = tuple(layer.name for layer in container.layers)
    for layer in container.layers:
        if weights.get(layer.name, _no_value) is not _no_value:
            continue
        if isinstance(layer, Container):
            get_keras_weights(layer, weights=weights)
        else:
            weights[layer.name] = layer.get_weights()
    return weights


def flatten_structure(structure, name, *,
                      skip_empty_weights=False,
                      skip_empty_layers=False):
    struct = structure[name]
    for lname in struct:
        value = structure[lname]
        if not isinstance(value, tuple):
            if (skip_empty_weights and value == []) or \
                    (skip_empty_layers and value.weights == []):
                continue
            yield lname
        else:
            yield from flatten_structure(structure, lname,
                                         skip_empty_weights=skip_empty_weights,
                                         skip_empty_layers=skip_empty_layers)


def set_keras_weights(container, weights, *, by_name=False, by_structure=False,
                      allow_missing=False, skip_empty=True, src_name=None):
    assert by_name+by_structure == 1, \
        ("One and only one of by_name and by_structure must be provided. Got "
         "{} of them.".format(by_name + by_structure))
    assert src_name is None or by_structure, \
        "Source name is only useful setting weights by structure."
    assert not allow_missing or by_name, \
        "Allowing missing layers can only happen when setting weights by name."
    cstruct = get_keras_structure(container)
    cleaves = flatten_structure(cstruct, container.name,
                                skip_empty_layers=skip_empty)
    if by_structure:
        if src_name is None:
            src_name = weights[""]
        wleaves = flatten_structure(weights, src_name,
                                    skip_empty_weights=skip_empty)
        mapping = {}
        for lname, wname in zip(cleaves, wleaves):
            current_target = mapping.get(lname, _no_value)
            if current_target is not _no_value:
                assert current_target == wname, \
                    ("Trying to associate {lname} with {wname}, but {lname} "
                     "is already associated with {current_target}"
                     .format(lname=lname, wname=wname,
                             current_target=current_target))
                continue
            mapping[lname] = wname
    if by_name:
        mapping = {}
        for lname in cleaves:
            mapping[lname] = lname

    for lname, wname in mapping.items():
        w = weights.get(wname, _no_value)
        if w is _no_value:
            if allow_missing:
                continue
            else:
                raise KeyError(wname)
        cstruct[lname].set_weights(w)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE |
                    doctest.ELLIPSIS)
