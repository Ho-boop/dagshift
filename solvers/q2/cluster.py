"""核亲和簇：把计算 op 沿大 tensor 边合并成工作量受限的簇，列表调度时同簇 op 绑定同一个核。

目的：小图上每条跨核边都要付 500 周期同步 + DDR 搬运（全机共享 60 B/cycle），逐 op 贪心选核会切出
大量跨核边；先按边的字节数做重边合并（heavy-edge），让“搬运代价高、计算量小”的局部留在一个核上。
簇不要求凸（逐 op 子图编码允许核间往返），只用于约束核归属；核内顺序仍由列表调度推导。
"""

from .model import scene_b


def edge_clusters(g, max_work, min_bytes=0, weight='bytes'):
    """并查集重边合并：按边权降序，合并后簇工作量 ≤ max_work 才合并。返回 {op: 簇代表}。

    weight='bytes'：边权 = tensor 字节；'ratio'：字节 / min(两端 op 周期)（搬运相对计算越贵越先合）。
    """
    sb = scene_b(g)
    parent = {u: u for u in g.compute_ids}
    work = {u: g.cycles[u] for u in g.compute_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    topo = g.topo_index
    for u in g.compute_ids:
        for tid, size, cons in sb.outs[u]:
            if size < min_bytes:
                continue
            for v in cons:
                if weight == 'ratio':
                    w = size / max(1, min(g.cycles[u], g.cycles[v]))
                else:
                    w = size
                edges.append((-w, topo[u], topo[v], u, v))
    edges.sort()
    for _, _, _, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if work[ru] + work[rv] > max_work:
            continue
        if work[ru] < work[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        work[ru] += work[rv]
    return {u: find(u) for u in g.compute_ids}
