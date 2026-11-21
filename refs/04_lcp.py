def lcp(words):
    if not words:
        return ""
    out = []
    for chars in zip(*words):
        if len(set(chars)) == 1:
            out.append(chars[0])
        else:
            break
    return "".join(out)