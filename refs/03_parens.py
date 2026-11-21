PAIRS = {")": "(", "]": "[", "}": "{"}
OPEN = set(PAIRS.values())


def check(s):
    stack = []
    for c in s:
        if c in OPEN:
            stack.append(c)
        elif c in PAIRS:
            if not stack or stack.pop() != PAIRS[c]:
                return False
        else:
            return False
    return not stack