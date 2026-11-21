def rle(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)


def unrle(s):
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        i += 1
        digits = ""
        while i < len(s) and s[i].isdigit():
            digits += s[i]
            i += 1
        if not digits:
            return ""
        out.append(ch * int(digits))
    return "".join(out)