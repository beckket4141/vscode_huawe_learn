import bisect
a = [(1,2), (3,5),(6,8)]
idx = bisect.bisect_right(a, (3,3))
print(a[idx])
print(a)
a.insert(idx, (2,2))
print(a)
a.remove(a[idx])
print(a)