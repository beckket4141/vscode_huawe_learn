import sys
sys.setrecursionlimit(10**7)

class Solution:
    def mindiv(self, target_list, rank_group):

        midiv = float('inf')
        bestgroup = []
        self.all_menthod(target_list, rank_group)
        for group in self.allmethod:
            curdiv = self.stdis(group)
            if curdiv < midiv:
                midiv = curdiv
                bestgroup = group
        return bestgroup, curdiv


    def all_menthod(self, target_list, rank_group): # 给出所有的分割子序列方法
        self.allmethod = []
        self.curmethod = []
        n = len(target_list)
        rest = rank_group

        def traverse(st, rest):
            rest -= 1
            if rest == 0:
                self.curmethod.append(target_list[st:])
                self.allmethod.append(self.curmethod[:])
                self.curmethod.pop()
                return

            for i in range(st+1, n-rest+1):
                self.curmethod.append(target_list[st:i])
                traverse(i, rest)
                self.curmethod.pop()

        traverse(0, rest)
    
    def stdis(self,nums):
        n = len(nums)
        sumlist = [0] * n
        for i in range(n):
            sumlist[i] = sum(nums[i])
        average = sum(sumlist)/n
        temp = 0
        for num in sumlist:
            temp += ((num - average)**2)/n
        res = temp**(0.5)
        return res
    
group, rank_group = map(int, input().strip().split())
target_list = list(map(int, input().strip().split()))

sol = Solution()
bs, res = sol.mindiv(target_list,rank_group)
resid = []
for i in range(rank_group):
    idx = len(bs[i])
    resid.append(idx)
print(" ".join(map(str, resid)))

#[[1, 2], [3], [4]]