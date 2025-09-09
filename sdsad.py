import sys
def parse_ipv4(bd):

    n = len(bd)
    bdn = [0] * n
    for i in range(n):
        s, e = bd[i][0], bd[i][1]
        sl = list(map(int, s.split("."))) 
        el = list(map(int, e.split(".")))
        sn = sl[0]<<24 | sl[1]<<16 | sl[2]<<8 |sl[3]
        en = el[0]<<24 | el[1]<<16 | el[2]<<8 |el[3]
        bdn[i] = [sn, en]
    return bdn

def recove_ipv4(bdn):
    final_bd = []
    for cur in bdn:
        s = ".".join(str((cur[0]>>8*i)&255) for i in range(3, -1, -1))
        e = ".".join(str((cur[1]>>8*i)&255) for i in range(3, -1, -1))
        final_bd.append([s,e])
    return final_bd

n = int(sys.stdin.readline().strip())
bd = [''] * n
for i in range(n):
    cur_bd = sys.stdin.readline().strip().strip('[]')
    bd[i] = cur_bd.split(',')
print(bd)
res = parse_ipv4(bd)
print(res)
res2 = recove_ipv4(res)
print(res2)