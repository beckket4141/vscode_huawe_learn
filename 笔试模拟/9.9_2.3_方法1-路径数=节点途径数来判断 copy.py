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
sys.setrecursionlimit(10**7)
class Solution:
    def single_wrong(self, adj, s, e):
        res = []
        n = len(adj)
        self.method_n = 0
        self.visited = [False]*n
        self.visited[s] = True
        self.curpath = []
        record = [0]*n
        def traverse(i):
            if i == e:
                self.method_n += 1
                for idx in self.curpath:
                    record[idx] += 1
                record[e] -= 1
                return
            for nex in adj[i]:
                if not self.visited[nex]:
                    self.visited[nex] = True
                    self.curpath.append(nex)
                    traverse(nex)
                    self.curpath.pop()
                    self.visited[nex] = False
        traverse(s)
        for i, path_num in enumerate(record):
            if path_num == self.method_n:
                res.append(i)
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
