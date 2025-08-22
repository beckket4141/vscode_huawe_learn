class Solution:
    def __init__(self):
        self.res = []
        self.book = {}

    def maxIter(self, versions):
        maxc = -1
        for ver in versions:
            count = self.dfs(ver, versions)
            if count > maxc:
                maxc = count
        b = list(self.book.items())
        for i in range(len(b)):
            if b[i][1] == maxc:
                self.res.append(b[i][0])
        self.res.sort()
        return self.res

    def dfs(self, ver, versions): 
        if ver in self.book:
            return self.book[ver]       
        if versions[ver] is None:
            self.book[ver] = 0
        else:
            self.book[ver] = self.dfs(versions[ver],versions)+1       
        return self.book[ver]



def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    versions = {}
    idx += 1
    for i in range(n):
        if data[idx+1] == "NA":
            versions[data[idx]] = None
        else:
            versions[data[idx]] = data[idx + 1]
        idx += 2

    so = Solution()
    print(" ".join(so.maxIter(versions)))


if __name__ == "__main__":
    main()