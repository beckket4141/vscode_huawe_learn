"""
5
1 2 3 0 1
2 3 1 0 2
输出
1
"""
import sys
from collections import deque

class Solution:
    def transTree(self, ini_tree, tar_tree):
        n = len(ini_tree)
        swich = 0
        self.res = 0
        def dfs(i_idx, swich):
            cur_color = ini_tree[i_idx]
            tar_color = tar_tree[i_idx]
            if cur_color == 0:
                return
            
            if swich != 0:
                cur_color = (cur_color + swich - 1 ) % 3 + 1
            
            if cur_color != tar_color:

                adds = ((tar_color - cur_color) + 3) % 3

                swich += adds
                self.res += adds
            if i_idx*2 + 1 < n:
                dfs(i_idx*2 + 1, swich)
            if i_idx*2 + 2 < n:
                dfs(i_idx*2 + 2, swich)
            

        dfs(0, swich)
        return self.res
                

n = int(input().strip())
ini_tree = list(map(int, sys.stdin.readline().strip().split()))
tar_tree = list(map(int, sys.stdin.readline().strip().split()))
# ini_root = build_tree(ini_tree)
# tar_root = build_tree(tar_tree)
sol = Solution()
print(sol.transTree(ini_tree, tar_tree))