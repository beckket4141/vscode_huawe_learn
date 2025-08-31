import sys
def main():
    input = sys.stdin.read
    data = list(map(int, input().split()))
    n, k = data[0], data[1]
    grid = [[True]*n for _ in range(n)]
    idx = 2
    for i in range(k):
        grid[data[idx]][data[idx+1]] = False
        idx += 2
    print(grid)

if __name__ == "__main__":
    main()