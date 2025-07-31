class CourseScheduleSolutions_207:
    """
    课程表问题解决方案集合
    https://leetcode.cn/problems/course-schedule-ii/
    """
    
    from typing import List

    class Solution1:
        from typing import List
        """
        解法一：DFS检测环（基础版本）
        """
        def __init__(self):
            # 记录递归堆栈中的节点
            self.onPath = []
            # 记录图中是否有环
            self.hasCycle = False

        def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
            graph = self.buildGraph(numCourses, prerequisites)
            
            self.onPath = [False] * numCourses
            
            for i in range(numCourses):
                # 遍历图中的所有节点
                self.traverse(graph, i)
            # 只要没有循环依赖可以完成所有课程
            return not self.hasCycle

        # 图遍历函数，遍历所有路径
        def traverse(self, graph: List[List[int]], s: int):
            if self.hasCycle:
                # 如果已经找到了环，也不用再遍历了
                return

            if self.onPath[s]:
                # s 已经在递归路径上，说明成环了
                self.hasCycle = True
                return
            
            # 前序代码位置
            self.onPath[s] = True
            for t in graph[s]:
                self.traverse(graph, t)
            # 后序代码位置
            self.onPath[s] = False
        
        ## 构建图
        @staticmethod
        def buildGraph(numCourses: int, prerequisites: List[List[int]]) -> List[List[int]]:
            # 图中共有 numCourses 个节点
            graph = [[] for _ in range(numCourses)]
            for edge in prerequisites:
                from_, to_ = edge[1], edge[0]
                # 添加一条从 from 指向 to 的有向边
                # 边的方向是「被依赖」关系，即修完课程 from 才能修课程 to
                graph[from_].append(to_)
            return graph

    class Solution2:
        from typing import List
        """
        解法二：DFS检测环（优化版本）- 使用visited数组避免重复遍历
        """
        def __init__(self):
            # 记录一次递归堆栈中的节点
            self.onPath = []
            # 记录节点是否被遍历过
            self.visited = []
            # 记录图中是否有环
            self.hasCycle = False

        def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
            graph = self.buildGraph(numCourses, prerequisites)
            
            self.onPath = [False] * numCourses
            self.visited = [False] * numCourses
            
            for i in range(numCourses):
                # 遍历图中的所有节点
                self.traverse(graph, i)
            # 只要没有循环依赖可以完成所有课程
            return not self.hasCycle

        # 图遍历函数，遍历所有路径
        def traverse(self, graph: List[List[int]], s: int):
            if self.hasCycle:
                # 如果已经找到了环，也不用再遍历了
                return

            if self.onPath[s]:
                # s 已经在递归路径上，说明成环了
                self.hasCycle = True
                return
            
            if self.visited[s]:
                # 不用再重复遍历已遍历过的节点
                return

            # 前序代码位置
            self.visited[s] = True
            self.onPath[s] = True
            for t in graph[s]:
                self.traverse(graph, t)
            # 后序代码位置
            self.onPath[s] = False
        
        ## 构建图
        @staticmethod
        def buildGraph(numCourses: int, prerequisites: List[List[int]]) -> List[List[int]]:
            # 图中共有 numCourses 个节点
            graph = [[] for _ in range(numCourses)]
            for edge in prerequisites:
                from_, to_ = edge[1], edge[0]
                # 添加一条从 from 指向 to 的有向边
                # 边的方向是「被依赖」关系，即修完课程 from 才能修课程 to
                graph[from_].append(to_)
            return graph

if __name__ == '__main__':
    numCourses = 2
    prerequisites = [[1,0]]
    
    # 测试解法一
    sol1 = CourseScheduleSolutions_207.Solution1()
    result1 = sol1.canFinish(numCourses, prerequisites)
    print(f"解法一结果: {result1}")
    
    # 测试解法二
    sol2 = CourseScheduleSolutions_207.Solution2()
    result2 = sol2.canFinish(numCourses, prerequisites)
    print(f"解法二结果: {result2}")
    
    # 测试有环的情况
    numCourses2 = 2
    prerequisites2 = [[1,0],[0,1]]
    
    result3 = sol1.canFinish(numCourses2, prerequisites2)
    print(f"解法一环检测结果: {result3}")
    
    result4 = sol2.canFinish(numCourses2, prerequisites2)
    print(f"解法二环检测结果: {result4}")