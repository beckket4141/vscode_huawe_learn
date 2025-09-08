"""
6
16 18 20 25 23 20
输出--<=4
1 2
4 5
"""
# 完成✅
n = int(input().strip())
nums = list(map(int, input().strip().split()))
maxres, maxstart = -1, []
def dfs(i):
    mi = ma = nums[i]
    res = 1
    for nx in range(i+1, n):
        cur = nums[nx]
        if not 18<= nums[nx] <=24:
            break
        if cur > ma:
            ma = cur
        elif cur < mi:
            mi = cur
        if ma - mi <= 4:
            res += 1
        else:
            break
    return res

for i in range(n):
    curres = -2
    if maxres > n-i:
        break
    if 18<= nums[i] <=24:
        curres = dfs(i)
    if curres >= maxres:
        if maxres == curres:
            maxstart.append(i)
        else:
            maxstart = [i]
        maxres = curres
        
if maxres == 0:
    print(0)
else:      
    for i in range(len(maxstart)):
        print(f"{maxstart[i]} {maxstart[i]+maxres-1}")
