def count(g):
    if not g:
        return 0
    seen = set()
    total = 0
    for i in range(len(g)):
        for j in range(len(g[i])):
            if g[i][j] != "1" or (i, j) in seen:
                continue
            total += 1
            stack = [(i, j)]
            seen.add((i, j))
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if (0 <= x < len(g) and 0 <= y < len(g[x])
                            and g[x][y] == "1" and (x, y) not in seen):
                        seen.add((x, y))
                        stack.append((x, y))
    return total