import sys, heapq
from collections import defaultdict
# Belady/OPT: on a miss at capacity, evict the resident key whose NEXT use is
# farthest in the future. Unimplementable online (needs the future) but it is the
# exact upper bound for ANY eviction policy on this trace, so it says how much
# room a smarter policy could possibly have.
path, cap = sys.argv[1], int(sys.argv[2])
acc = []
for line in open(path):
    demand = line.split('|')[0].split()
    acc.extend(int(t) for t in demand)
n = len(acc)
# next_use[i] = index of the next access to the same key after i (n = never again)
nxt = [n]*n
last = {}
for i in range(n-1, -1, -1):
    k = acc[i]
    nxt[i] = last.get(k, n)
    last[k] = i
resident = set()
heap = []                      # (-next_use, key) lazy max-heap on next use
cur = {}                       # key -> its currently-recorded next use
hits = misses = 0
for i, k in enumerate(acc):
    if k in resident:
        hits += 1
    else:
        misses += 1
        if len(resident) >= cap:
            while True:        # pop stale entries (key evicted, or next_use updated)
                negu, vk = heapq.heappop(heap)
                if vk in resident and cur.get(vk) == -negu:
                    resident.discard(vk); break
        resident.add(k)
    cur[k] = nxt[i]
    heapq.heappush(heap, (-nxt[i], k))
print(f"  accesses {n}  unique {len(last)}  cap {cap}")
print(f"  OPT (Belady): hit {100.0*hits/n:.2f}%   miss {100.0*misses/n:.2f}%  ({misses/512:.0f} miss/tok)")
