class AllInOneSegmentTree:
    class SegmentNode:
        def __init__(self, l, r, merge_value):
            self.l = l
            self.r = r
            self.merge_value = merge_value
            self.lazy_add = 0
            self.lazy_assign = 0
            self.has_lazy_assign = False
            self.left = None
            self.right = None

    def __init__(self, start, end, default_value, tree_type):
        if tree_type not in ["sum", "max", "min"]:
            raise ValueError("Invalid type, must be sum, max, or min")
        self.tree_type = tree_type
        self.default_value = default_value
        root_merge_value = self.compute_range_value(start, end, default_value)
        self.root = self.SegmentNode(start, end, root_merge_value)

    # 计算区间 [l, r] 赋值为 val 后的 mergeValue
    def compute_range_value(self, l, r, val):
        # 如果类型为求和，则返回区间长度乘以 val
        if self.tree_type == "sum":
            return (r - l + 1) * val
        else:
            # 如果类型为求最大值或最小值，则返回 val
            return val

    # 根据区间长度更新 mergeValue，加上 delta
    def apply_add_to_value(self, node, delta):
        if self.tree_type == "sum":
            return node.merge_value + (node.r - node.l + 1) * delta
        else:
            # 如果类型为求最大值或最小值，则返回当前值加上 delta
            return node.merge_value + delta

    # 根据类型合并左右子节点的值
    def merge(self, left_val, right_val):
        if self.tree_type == "sum":
            return left_val + right_val
        elif self.tree_type == "max":
            return max(left_val, right_val)
        elif self.tree_type == "min":
            return min(left_val, right_val)
        raise ValueError("Invalid type")

    # 动态创建线段树节点
    def init_children_if_needed(self, node):
        if node.l == node.r:
            return
        mid = node.l + (node.r - node.l) // 2
        if node.left is None:
            left_merge_value = self.compute_range_value(node.l, mid, self.default_value)
            node.left = self.SegmentNode(node.l, mid, left_merge_value)
        if node.right is None:
            right_merge_value = self.compute_range_value(mid + 1, node.r, self.default_value)
            node.right = self.SegmentNode(mid + 1, node.r, right_merge_value)

    # 下传懒标记，保证子节点的数据正确
    def push_down(self, node):
        # 如果存在赋值懒标记，优先下传赋值
        if node.has_lazy_assign:
            self.apply_assign(node.left, node.lazy_assign)
            self.apply_assign(node.right, node.lazy_assign)
            node.has_lazy_assign = False
            node.lazy_assign = 0
        # 下传累加懒标记
        if node.lazy_add != 0:
            self.apply_add(node.left, node.lazy_add)
            self.apply_add(node.right, node.lazy_add)
            node.lazy_add = 0

    # 将赋值懒标记下传到子节点
    def apply_assign(self, child, val):
        child.has_lazy_assign = True
        child.lazy_assign = val
        # 清除累加懒标记
        child.lazy_add = 0
        child.merge_value = self.compute_range_value(child.l, child.r, val)

    # 将累加懒标记下传到子节点
    def apply_add(self, child, delta):
        if child.has_lazy_assign:
            # 如果子节点已有赋值懒标记，则直接更新该赋值
            child.lazy_assign += delta
            child.merge_value = self.compute_range_value(child.l, child.r, child.lazy_assign)
        else:
            # 如果子节点没有赋值懒标记，则更新累加懒标记
            child.lazy_add += delta
            child.merge_value = self.apply_add_to_value(child, delta)

    # 单点赋值：将索引 index 赋值为 val
    def update(self, index, val):
        # 直接复用区间赋值方法
        self.range_update(index, index, val)

    # 区间赋值：将闭区间 [qL, qR] 赋值为 val
    def range_update(self, qL, qR, val):
        self._range_update(self.root, qL, qR, val)

    def _range_update(self, node, qL, qR, val):
        if node.r < qL or node.l > qR:
            raise ValueError("Invalid update range")
        # 当前节点完全包含于更新区间内
        if qL <= node.l and node.r <= qR:
            node.has_lazy_assign = True
            node.lazy_assign = val
            node.lazy_add = 0
            node.merge_value = self.compute_range_value(node.l, node.r, val)
            return

        self.init_children_if_needed(node)
        self.push_down(node)

        mid = node.l + (node.r - node.l) // 2
        if qR <= mid:
            self._range_update(node.left, qL, qR, val)
        elif qL > mid:
            self._range_update(node.right, qL, qR, val)
        else:
            self._range_update(node.left, qL, mid, val)
            self._range_update(node.right, mid + 1, qR, val)
        node.merge_value = self.merge(node.left.merge_value, node.right.merge_value)

    # 单点累加：将索引 index 增加 delta（可为负数）
    def add(self, index, delta):
        # 直接复用区间累加方法
        self.range_add(index, index, delta)

    # 区间累加：将闭区间 [qL, qR] 增加 delta（可为负数）
    def range_add(self, qL, qR, delta):
        self._range_add(self.root, qL, qR, delta)

    def _range_add(self, node, qL, qR, delta):
        if node.r < qL or node.l > qR:
            raise ValueError("Invalid update range")
        if qL <= node.l and node.r <= qR:
            if node.has_lazy_assign:
                # 若已有赋值懒标记，则更新赋值值
                node.lazy_assign += delta
                node.merge_value = self.compute_range_value(node.l, node.r, node.lazy_assign)
            else:
                node.lazy_add += delta
                node.merge_value = self.apply_add_to_value(node, delta)
            return
        self.init_children_if_needed(node)
        self.push_down(node)

        mid = node.l + (node.r - node.l) // 2
        if qL <= mid:
            self._range_add(node.left, qL, qR, delta)
        if qR > mid:
            self._range_add(node.right, qL, qR, delta)
        node.merge_value = self.merge(node.left.merge_value, node.right.merge_value)

    # 查询闭区间 [qL, qR] 的聚合值
    def query(self, qL, qR):
        return self._query(self.root, qL, qR)

    def _query(self, node, qL, qR):
        if node.r < qL or node.l > qR:
            raise ValueError("Invalid query range")
        if qL <= node.l and node.r <= qR:
            return node.merge_value

        self.init_children_if_needed(node)
        self.push_down(node)

        mid = node.l + (node.r - node.l) // 2
        if qR <= mid:
            return self._query(node.left, qL, qR)
        elif qL > mid:
            return self._query(node.right, qL, qR)
        else:
            left_result = self._query(node.left, qL, mid)
            right_result = self._query(node.right, mid + 1, qR)
            return self.merge(left_result, right_result)