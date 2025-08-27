import sys
def main():
    m, n = map(int,sys.stdin.readline().split())
    grid = [[] for _ in range(m)]
    for i in range(m):
        grid[i] = list(map(int,sys.stdin.readline().split()))

    print(grid)
if __name__ == '__main__':
    main()