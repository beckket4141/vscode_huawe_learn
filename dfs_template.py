def dfs(node, visited):
    """
    DFS算法框架模板
    
    Args:
        node: 当前访问的节点
        visited: 已访问节点的集合或列表
    
    框架结构：
    1. 结束条件判断
    2. 做选择（标记当前节点为已访问）
    3. 遍历当前节点的邻居节点
    4. 递归调用
    5. 撤销选择（根据需要）
    """
    # 结束条件：到达叶子节点
    if is_leaf_node(node):
        return
    
    # 做选择：标记当前节点为已访问
    visited.add(node)
    
    # 遍历当前节点的所有邻居
    for neighbor in get_neighbors(node):
        # 检查邻居节点是否已访问
        if neighbor not in visited:
            # 递归访问邻居节点
            dfs(neighbor, visited)
    
    # 撤销选择（根据需要，有些问题需要，有些不需要）
    # visited.remove(node)  # 如果需要回溯


def is_leaf_node(node):
    """
    判断是否为叶节点的辅助函数
    根据具体问题实现
    """
    pass


def get_neighbors(node):
    """
    获取节点邻居的辅助函数
    根据具体问题实现
    """
    pass