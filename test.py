
def main():
    n, m = map(int, input().strip().split())
    nums = []
    for i in range(m):
        geshu, num = map(int, input().strip().split())
        nums.append((geshu, num))
    print(nums)

if __name__ == "__main__":
    main()