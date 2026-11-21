import ast


def parse(s):
    # json.loads rejects Python's None/True/False, so rewrite the JSON literals
    # into their Python spelling and let ast evaluate them as a pure literal
    # (no names, no calls, no attribute access can run).
    for word, val in (("true", True), ("false", False), ("null", None)):
        s = s.replace(word, repr(val))
    return ast.literal_eval(s.strip())