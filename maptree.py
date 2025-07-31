from typing import List
## 构建图
def buildGraph(numCourses: int, prerequisites: List[List[int]]) -> List[List[int]]:
    # 图中共有 numCourses 个节点
    graph = [[] for _ in range(numCourses)]
    for edge in prerequisites:
        from_, to_ = edge[1], edge[0]
        # 添加一条从 from 指向 to 的有向边
        # 边的方向是「被依赖」关系，即修完课程 from 才能修课程 to
        graph[from_].append(to_)
    return graph

# 测试样例
if __name__ == '__main__':
    numCourses = 2
    prerequisites = [[1,0],[0,0],[0,1], [1,1], [2,0]]
    print(buildGraph(numCourses, prerequisites))