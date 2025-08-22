def backtrack(path, choices):
    """
    回溯算法框架模板
    
    Args:
        path: 已经做出的选择列表（路径）
        choices: 当前可做的选择列表
    
    框架结构：
    1. 结束条件判断
    2. 遍历选择列表
    3. 做选择
    4. 递归调用
    5. 撤销选择
    """
    # 结束条件：到达叶子节点
    if not choices or meet_end_condition():
        # 处理结果
        result.append(path[:])  # 添加路径副本
        return

    # 遍历选择列表
    for i, choice in enumerate(choices):
        # 做选择
        path.append(choice)
        new_choices = choices[:i] + choices[i+1:]  # 移除已选择的元素
        
        # 进入下一层决策树
        backtrack(path, new_choices)
        
        # 撤销选择（回溯）
        path.pop()


def meet_end_condition():
    """
    判断是否满足结束条件的辅助函数
    根据具体问题实现
    """
    pass


# 全局结果列表
result = []