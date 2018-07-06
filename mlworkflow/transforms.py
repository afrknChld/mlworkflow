class GenericTransform:
    def __init__(self, signature):
        sigs = signature.split()
        self.transforms = [getattr(self, sig)
                           for sig in sigs]

    def none(self, item):
        return item

    def image(self, item, init=False):
        pass

    def __call__(self, item):
        elem_transforms = zip(item, self.transforms)
        e, t = next(elem_transforms)
        result = [t(e, init=True)]
        for e, t in elem_transforms:
            result.append(t(e))
        return result
