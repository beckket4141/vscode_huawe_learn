import sys
from collections import deque
class Solution:
    def mindistance(self,grid):
        m, n = len(grid), len(grid[0])
        start = []
        des = []
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    start.append((i,j))
                elif grid[i][j] == 0:
                    des.append((i,j))
        for i in range(len(start)):
            distance = self.bfs(grid,start[i])
            if distance > 0:
                res += self.bfs(grid,start[i])
        return res

    def bfs(self, grid, start):
        m, n =len(grid), len(grid[0])
        visited = [[-1]*n for _ in range(m)]
        visited[start[0]][start[1]] = 0
        q = deque()
        q.append(start)
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0<=nx<m and 0<=ny<n and visited[nx][ny] == -1 and grid[nx][ny] != -1:
                    if grid[nx][ny] == 0:
                        return visited[x][y]+1
                    visited[nx][ny] = visited[x][y]+1
        return 0

if __name__ == "__main__":
    input = sys.stdin.readline
    m, n = map(int, input().strip().split())
    grid = [[] for _ in range(m)]
    for i in range(m):
        grid[i] = list(map(int, input().strip().split()))
    solution = Solution()
    print(solution.mindistance(grid))
    
        