"""
13                      
check
in c1
in c2
count
check
in b3
in b4
count
check
out
check
out
check
"""
import heapq
import sys

data = list(sys.stdin.read().strip().split())
n = int(data[0])
dic_pq = set()
pq = []
q = []
idx = 1
for i in range(n):
    op = data[idx]

    if op == 'check':
        while pq:
            cur = pq[0]
            if cur in dic_pq:
                break
            else:
                heapq.heappop(pq)
        
        print(cur if pq else 'EMPTY')
        idx += 1

    elif op == 'in':
        ob = data[idx+1]
        dic_pq.add(ob)
        heapq.heappush(pq, ob)
        q.append(ob)
        idx += 2

    elif op == 'count':
        print(len(q))
        idx += 1

    elif op == 'out':
        if not q:
            print('EMPTY')
            idx += 1
            continue
        cur = q.pop()
        print(cur)
        dic_pq.remove(cur)
        idx += 1
    