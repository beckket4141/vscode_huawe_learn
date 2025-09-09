"""
3
[192.168.1.1,192.168.1.3]
[192.168.1.2,192.168.1.3]
[192.168.1.4,192.168.1.5]
输出
[192.168.1.1,192.168.1.5]
"""
import sys
class Solution:
    def mergeipv4(self, bd):
        n = len(bd)
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
        
        bdn = parse_ipv4(bd)
        bdn.sort()
        f, l = bdn[0][0], bdn[0][1]
        final_bdn = []
        for i in range(n):
            fn, ln = bdn[i][0], bdn[i][1]
            if fn <= l+1:
                l = max(ln, l)
            else:
                final_bdn.append([f, l])
                f, l = fn, ln
        final_bdn.append([f, l])

        return recove_ipv4(final_bdn)

n = int(sys.stdin.readline().strip())
bd = [''] * n
for i in range(n):
    cur_bd = sys.stdin.readline().strip().strip('[]')
    bd[i] = cur_bd.split(',')
sol = Solution()
res = sol.mergeipv4(bd)
print(" ".join(f"[{res[i][0]},{res[i][1]}]" for i in range(len(res))))


