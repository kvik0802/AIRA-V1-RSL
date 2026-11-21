class N:
    def __init__(s, v, n=None):
        s.v, s.n = v, n


def solve(h):
    vals = []
    while h:
        vals.append(h.v)
        h = h.n
    out = None
    for v in vals:
        out = N(v, out)
    return out