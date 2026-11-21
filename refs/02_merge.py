class N:
    def __init__(s, v, n=None):
        s.v, s.n = v, n


def merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    out = None
    tail = None
    while a or b:
        if b is None or (a is not None and a.v <= b.v):
            v, a = a.v, a.n
        else:
            v, b = b.v, b.n
        node = N(v)
        if out is None:
            out = node
        else:
            tail.n = node
        tail = node
    return out