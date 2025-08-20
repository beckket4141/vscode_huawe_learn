def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    m, n = int(data[0]), int(data[1])
    grid = [[] for _ in range(m)]
    idx = 2
    hole = []
    for i in range(m):
        row = data[idx]
        grid[i] = list("".join(row))
        idx += 1

    print(grid)
if __name__ == "__main__":
    main()