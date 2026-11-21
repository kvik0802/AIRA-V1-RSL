import heapq


class Kth:
    """Keeps the k largest values seen so far in a min-heap, so min() of the
    heap is the k-th largest."""

    def __init__(s, k):
        s.k = k
        s.h = []

    def add(s, v):
        heapq.heappush(s.h, v)
        if len(s.h) > s.k:
            heapq.heappop(s.h)
        return s.h[0]