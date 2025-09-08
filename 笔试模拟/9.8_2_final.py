"""
3 2
1 2 3 
"""
import math
import sys
sys.setrecursionlimit(10**7)

class Solution:
    def mindiv(self, origin_nums, k):
        n = len(origin_nums)
        average = sum(origin_nums)/k
        presum = [0]*(n+1)
        for i in range(1, n+1):
            presum[i] =  presum[i-1] + origin_nums[i-1]   

        self.cur_res = []
        self.best_res = []
        self.best_div = float('inf')
        def traverse(start, rest):
            rest -= 1
            if rest == 0:
                self.cur_res.append(n-start)
                #print(self.cur_res)
                cur_div = float('inf')
                temp = 0
                idx = 0
                for i in range(k):
                    cursum = presum[idx+self.cur_res[i]] - presum[idx]
                    #print(curlist)
                    temp += ((cursum-average)**2)/k
                    idx += self.cur_res[i]
                cur_div = math.sqrt(temp)
                #print(cur_div)
                #print(cur_div)
                #print(self.cur_res)

                if cur_div < self.best_div:
                    self.best_res = self.cur_res[:]
                    self.best_div = cur_div

                self.cur_res.pop()
                return
             
            
            for i in range(start+1, n-rest+1):
                self.cur_res.append(i - start)
                traverse(i, rest)
                self.cur_res.pop()

        traverse(0, k)
        #print(self.best_div)
        return self.best_res
    


n, k = map(int, input().strip().split())
origin_nums = list(map(int, input().strip().split()))

sol = Solution()
res = sol.mindiv(origin_nums, k)
print(" ".join(map(str, res)))