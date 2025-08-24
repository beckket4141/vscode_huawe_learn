import sys
sys.setrecursionlimit(1000000000)
def main():
    input = sys.stdin.read
    data = list(input().split())
    m, n = int(data[0]), int(data[1])
    idx = 2
    grid = [[""]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            grid[i][j] = data[idx]
            idx += 1
    target = data[idx]
    print(grid)
    print(target)

if __name__ == "__main__":
    main()