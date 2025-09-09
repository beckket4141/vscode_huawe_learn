"""
2
C01M23B050130 C01M23B060130
输出---一个大写字母+两个数字
C01,2
M23,2
B05,1;B06,1
"""
from collections import defaultdict
import sys

n = int(sys.stdin.readline().strip())
targets_list = list(sys.stdin.readline().strip().split())
cpu_dic = {}
me_dic = {}
bo_dic = {}

for target in targets_list:
    c = m = b = 1
    l = len(target)
    idx = 0
    while idx < l-2:
        cur = target[idx]
        n1, n2 = target[idx+1], target[idx+2]
        if cur in 'CMB' and n1 in '0123456789' and n2 in '0123456789':
            id = cur + n1 + n2
            if cur == 'C' and c == 1 :
                if id not in cpu_dic:
                    cpu_dic[id] = 0
                cpu_dic[id] += 1
                idx += 2
                c -= 1
            elif cur == 'M' and m == 1:
                if id not in me_dic:
                    me_dic[id] = 0
                me_dic[id] += 1
                idx += 2
                m -= 1
            elif cur == 'B' and b == 1:
                if id not in bo_dic:
                    bo_dic[id] = 0
                bo_dic[id] += 1
                idx += 2
                b -= 1
        idx += 1

clist = list(cpu_dic.items())
mlist = list(me_dic.items())
blist = list(bo_dic.items())
clist.sort()
mlist.sort()
blist.sort()
res = ''
for i in range(len(clist)):
    res += clist[i][0] + ','+ str(clist[i][1]) + ';'
print(res.strip(';'))
res = ''
for i in range(len(mlist)):
    res += mlist[i][0] + ','+ str(mlist[i][1]) + ';'
print(res.strip(';'))
res = ''
for i in range(len(blist)):
    res += blist[i][0] + ','+ str(blist[i][1]) + ';'
print(res.strip(';'))

#print(','.join(c for c in clist) )

            
 
                