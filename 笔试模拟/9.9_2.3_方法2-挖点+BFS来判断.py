"""
4
0 1 0 0
1 0 1 0
0 1 0 1
0 0 1 0
0 3
输出
1 2
"""
import sys
from collections import deque
sys.setrecursionlimit(10**7)
class Solution:
    def single_wrong(self, adj, s, e):
        res = []
        n = len(adj)
        for i in range(n):
            if i == s or i == e:
                continue
            self.visited = [False]*n
            self.visited[i] = True
            res.append(i)
            q = deque()
            q.append(s)
            found = False
            while q:
                cur = q.popleft()   
                for neb in adj[cur]:
                    if not self.visited[neb]:
                        if neb == e:
                            res.pop()
                            found = True
                            break
                        q.append(neb)
                        self.visited[neb] = True
                if found: break

        return res

                    


n = int(sys.stdin.readline().strip())
adj = [[] for _ in range(n)]
for i in range(n):
    line = list(sys.stdin.readline().strip().split())
    for j in range(n):
        if line[j] == '1':
            adj[i].append(j)

s, e = map(int, sys.stdin.readline().strip().split())

sol = Solution()
res = sol.single_wrong(adj, s, e)
print(' '.join(map(str, res)))
