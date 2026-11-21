def mpow(a, n):
    def mul(x, y):
        return [[sum(x[i][k] * y[k][j] for k in range(len(y)))
                 for j in range(len(y[0]))] for i in range(len(x))]

    size = len(a)
    r = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    while n:
        if n & 1:
            r = mul(r, a)
        a = mul(a, a)
        n >>= 1
    return r