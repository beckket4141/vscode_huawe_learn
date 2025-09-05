

n = int(input().strip())
areas = [[] for _ in range(n)]
for i in range(n):
    s, e = map(str, input().strip("[]").split(","))
    sl, el = list(map(int, s.split("."))), list(map(int, e.split(".")))
    areas[i] = [sl, el]

areas.sort(key = lambda x : (x[0][0], x[0][1], x[0][2], x[0][3]) )
print(areas)