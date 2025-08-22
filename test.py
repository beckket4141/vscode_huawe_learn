a = {}
a["i"] = 1
a["j"] = 2
b = list(a.items())
b.sort(key=lambda x: x[1], reverse=True)
print(b)