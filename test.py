import sys
def main():
    m, n = map(int, sys.stdin.readline().split())
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(m)]


    print(grid)
if __name__ == '__main__':
    main()
