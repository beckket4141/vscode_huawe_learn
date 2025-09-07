import sys

def parse(data):
    segs = []
    datalist = list(map(str, data.split(",")))
    for curdata in datalist:
        if "-" in curdata:
            s, e = map(int, curdata.split("-"))
        else:
            s = e = int(curdata)
        segs.append((s, e))
    return segs

def merge(segs):
    if not segs:
        return []
    segs.sort()
    res = [segs[0]]
    for ns, ne in segs[1:]:
        s, e = res[-1][0], res[-1][1]
        if ns <= e+1:
            res[-1] = (s, max(ne, e))
        else:
            res.append((ns, ne))
    return res

def remove_segs(bd, dels):
    cur = bd
    dels = merge(dels)
    for ds, de in dels:
        temp = []
        for s, e in cur:
            if s > de or e < ds:
                temp.append((s, e))
            else:
                if s < ds:
                    temp.append((s, ds-1))
                if de < e:
                    temp.append((de+1, e))
        cur = temp
    return merge(cur)


def main():
    n = int(sys.stdin.readline().strip())
    bd = []
    for _ in range(n):
        line = list(map(str, sys.stdin.readline().strip().split()))
        
        if line[0] == "undo":
            data = line[2]
            segs = parse(data)
            # print(segs)
            # print(bd)
            if bd:
                bd = remove_segs(bd, segs)

        else:
            data = line[1]
            segs = parse(data)
            bd.extend(segs)
            bd = merge(bd)
    if not bd:
        print(0)
    else:
        print(",".join(f"{s}-{e}" if s != e else str(s) for s, e in bd))


if __name__ == "__main__":
    main()