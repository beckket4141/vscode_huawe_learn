"""
4
2 5 1 8
输出
4 6
"""
import heapq
import sys

class Solution:
    def pk_winer(self, nums):
        n = len(nums)
        maxn = 2**31 - 1
        pkmenbers = [(nums[i], i) for i in range(n)]
        heapq.heapify(pkmenbers)
        while pkmenbers and len(pkmenbers) > 1:
            a = heapq.heappop(pkmenbers)
            b = heapq.heappop(pkmenbers)
            if a[0] == b[0]:
                continue
            elif a[0] < b[0]:
                winer_life = (b[0]-a[0])*3
                winer_id = b[1]
            else: 
                winer_life = (a[0]-b[0])*3
                winer_id = a[1]
            if winer_life > maxn:
                winer_life = maxn
            heapq.heappush(pkmenbers, (winer_life, winer_id))
        return (pkmenbers[0][1]+1 , pkmenbers[0][0]) if pkmenbers else ()

        

n = int(sys.stdin.readline().strip())
nums = list(map(int, sys.stdin.readline().strip().split()))
sol = Solution()
res = sol.pk_winer(nums)
print(" ".join(map(str, res)) if len(res) == 2 else -1)
