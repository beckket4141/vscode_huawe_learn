"""
2
5
aprint
bprint
aaprint
bbprint
output
print

输出:
aprint bprint aaprint bbprint
"""
import functools
import sys
sys.setrecursionlimit(10**7)

class Solution:
    def close_dic(self, s, t_list, limit_dis):
        res = []
        for t in t_list:
            cur_dis = self.mindistance(s,t)
            if cur_dis <= limit_dis:
                res.append((cur_dis, t))
        res.sort()
        return res


    def mindistance(self, s, t):
        ns, nt = len(s), len(t)

        @functools.cache
        def dp(s, si, t, tj):
            if si == -1:
                return tj+1
            if tj == -1:
                return si+1
            
            if s[si] == t[tj]:
                return dp(s, si-1, t, tj-1)
            else:
                op_del = dp(s, si-1, t, tj) + 1
                op_insert = dp(s, si, t, tj-1) + 1
                op_change = dp(s, si-1, t, tj-1) + 1
                return min(op_del, op_insert, op_change)
        res = dp(s, ns-1, t, nt-1)
        return res
    
limit_dis = int(sys.stdin.readline().strip())
n = int(sys.stdin.readline().strip())
t_list = []
for i in range(n):
    t = sys.stdin.readline().strip()
    t_list.append(t)
s = sys.stdin.readline().strip()

sol = Solution()
res = sol.close_dic(s, t_list, limit_dis)
print(" ".join(cs for _ , cs in res) if len(res) != 0 else None) 

# s = "app"
# t = "apple"
# sol = Solution()
# print(sol.closedic(s,[t],3))
    
