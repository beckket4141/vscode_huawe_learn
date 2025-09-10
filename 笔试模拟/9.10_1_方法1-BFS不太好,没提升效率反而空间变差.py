"""
5 5
10
3
10 2 3 4 5
1 2 3 4 10
1 2 3 10 5
1 10 3 4 5
1 2 3 4 5
输出
3 2 1 3 4 1
"""
from collections import deque
import sys

class Solution:
    def close_m(self, grid, target, need_n):
        all_y, all_x = len(grid), len(grid[0])
        mid_y, mid_x = (all_y-1)//2, (all_x-1)//2
        q = deque()
        q.append((mid_y, mid_x))
        record = []
        self.visited = [[-1]*all_x for _ in range(all_y)]
        self.visited[mid_y][mid_x] = 0
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        while q:
            y, x = q.popleft()
            if grid[y][x] == target:
                    need_n -= 1
                    record.append((self.visited[y][x], x, y))
            for dy, dx in directions:
                ny, nx = y+dy, x+dx
                if 0 <= ny < all_y and 0 <= nx < all_x and self.visited[ny][nx] == -1:
                    q.append((ny, nx))
                    self.visited[ny][nx] = self.visited[y][x] + 1
        return record
 
w, h = map(int, sys.stdin.readline().strip().split())
target = int(sys.stdin.readline().strip())
need_n = int(sys.stdin.readline().strip())
grid = [[] for _ in range(h)]
for i in range(h): # y 对应h , x 对应w
    grid[i] = list(map(int, sys.stdin.readline().strip().split()))

sol = Solution()
res = sol.close_m(grid, target, need_n)
res.sort()
print(" ".join(f"{res[i][1]} {res[i][2]}" for i in range(min(need_n, len(res)))))
