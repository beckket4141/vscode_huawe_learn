import sys
def main():
    input = sys.stdin.read
    data = list(map(int, input().split()))
    seatn, stan, usern = data[0], data[1], data[2]
    idx = 3
    userbook = [[]for _ in range(usern)]
    for i in range(usern):
        up, down = data[idx], data[idx+1]
        userbook[i].append((up,down))
        idx += 2
    print(userbook)

if __name__ == "__main__":
    main()