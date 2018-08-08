class GenericTransform:
    def __init__(self, signature):
        if not signature.startswith(">"):
            signature = ">" + signature
        sigs = signature.split()
        self.transforms = [getattr(self, sig)
                           for sig in sigs]

    def none(self, item):
        return item

    def __call__(self, item):
        return [t(e) for e, t in zip(item, self.transforms)]

    def __getattr__(self, name):
        if name.startswith(">"):
            f = getattr(self, name[1:])
            return lambda item: f(item, init=True)
        raise AttributeError(name)
