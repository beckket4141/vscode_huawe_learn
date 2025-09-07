"""
5
node1 15
node2 12
node3 13
node4 4
node5 50
3
node1 node2
node3 node2
node4 node5
输出
node5 54
"""
import sys
class Solution:
    def maxnodenet_maxnode(self, adj, weights):
        n = len(weights)
        visited = [False] * n
        maxsum = -1

        def dfs(i):
            visited[i] = True
            if weights[i] > weights[self.mweightn]:
                self.mweightn = i
            res = weights[i]
            for idx in adj[i]:
                if not visited[idx]:
                    res += dfs(idx)      
            return res

                
        for i in range(n):
            if not visited[i]:
                self.mweightn = i
                cursum = dfs(i)
                if cursum > maxsum:
                    maxsum = cursum
                    maxnode = self.mweightn
        return maxnode, maxsum
            
         


def main():
    input = sys.stdin.read
    data = list(map(str, input().split()))
    n = int(data[0])
    name_to_idx = {}
    idx_to_name = {}
    idx = 1
    weights = [-1]*n
    for i in range(n):
        nodename, weight =  data[idx], int(data[idx+1])
        weights[i] = weight
        name_to_idx[nodename] = i
        idx_to_name[i] = nodename
        idx += 2
    m = int(data[idx])
    idx += 1
    adj = [[] for _ in range(n)]
    for i in range(m):
        nodename1, nodename2 = data[idx], data[idx+1]
        idx1, idx2 = name_to_idx[nodename1], name_to_idx[nodename2]
        try:
            adj[idx1].append(idx2)
            adj[idx2].append(idx1)
        except:
             print(idx1,idx2)
             print(adj)
        idx += 2

    sol = Solution()
    maxnode, maxweight = sol.maxnodenet_maxnode(adj, weights)
    maxnodename = idx_to_name[maxnode]
    print(str(maxnodename)+ " " + str(maxweight))


if __name__ == "__main__":
        main()