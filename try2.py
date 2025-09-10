class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
    
a = TreeNode(1)
a.next = TreeNode(2)
print(a)
