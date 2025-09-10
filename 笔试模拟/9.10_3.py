"""
0 2
6
0 1 2
0 5 3
5 3 3
1 2 3
2 4 10
1 3 4
输出
6
0 1 3
"""
import sys
sys.setrecursionlimit(10**7)

data = list(map(int, sys.stdin.read().strip().split()))
s, t = data[0], data[1]
m = data[2]
idx = 3
mp = {}
for i in range(m):
    curs = data[idx]
    end = data[idx+1]
    money = data[idx+2]
    mp[(curs, end)] = mp.get((curs, end), 0) + money

nodes = set()
for (curs, end) in mp:
    nodes.add(curs); nodes.add(end)