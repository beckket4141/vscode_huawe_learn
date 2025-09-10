"""
5
1 2 3 0 1
2 3 1 0 2
输出
1
"""
import sys
from collections import deque
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def transTree(self, ini_root, tar_root):
        swich = 0
        self.res = 0
        def dfs(ini_node, tar_node, swich):
            if not ini_node:
                return
            
            if swich != 0:
                ini_node.val = (ini_node.val + swich -1 ) % 3 + 1
            
            if ini_node.val != tar_node.val:
                # print("ok")
                # print(tar_node.val, ini_node.val)
                adds = ((tar_node.val - ini_node.val) + 3) % 3
                # print(adds)
                swich += adds
                self.res += adds

            dfs(ini_node.left, tar_node.left, swich)
            dfs(ini_node.right, tar_node.right, swich)
            

        dfs(ini_root, tar_root, swich)
        return self.res
                

        

def build_tree(nums):
    n = len(nums)
    if nums is None or nums[0] == 0:
        return None
    root = TreeNode(nums[0])
    q = deque()
    q.append(root)
    idx = 1
    while q :
        curnode = q.popleft()
        #print(curnode.val)
        if  idx < n and nums[idx] != 0:
            curnode.left = TreeNode(nums[idx])
            q.append(curnode.left)
        if idx+1 < n and nums[idx + 1] != 0:
            curnode.right = TreeNode(nums[idx+1])
            q.append(curnode.right)
        idx += 2
    return root

n = int(input().strip())
ini_tree = list(map(int, sys.stdin.readline().strip().split()))
tar_tree = list(map(int, sys.stdin.readline().strip().split()))
ini_root = build_tree(ini_tree)
tar_root = build_tree(tar_tree)
sol = Solution()
print(sol.transTree(ini_root, tar_root))