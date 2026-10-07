"""问题3 solver 入口：问题2 的候选生成与搜索 + 用问题3（只读 FIFO Cache）评估器择优。

solve(graph, k, problem=3, seed=0, budget_s=None, **opts) -> plan
- 热启动：最优档案里问题3、问题2、问题1 同 k 的最优方案，以及问题3、问题2 在 k−1 核下的最优方案补一个空核，
  一律按问题3 重新评估（q2.archive_candidates）；
- 其余与问题2 solver 相同（块表示重排、列表调度候选组合、块搜索与列表调度邻域搜索），但全部用问题3 评估器打分；
- 恒不比保底基线差：保底基线两变体总在候选里；
- 跑批必须加 --pair：同一方案再用问题2 评估一次，得到 T_noL2 / T_cache 硬件对照。
Cache 感知的列表调度（listsched 的 cache=True）单独用没有平均收益，只作为多样性补进组合末尾
（grid3，见 solvers/q3/ALGORITHM.md）。
v2.2：继承问题2 v2.2 的关键链引导块搜索（crit_search），关键链按问题3 评估器的逐事件时间线求出。
v2.3：继承问题2 v2.3，关键链重算复用这一步的完整评估结果（reuse_full），省掉每次改进后的一次完整评估。
"""

from solvers.q2 import main as q2

VERSION = 'q3-from-q2-v2.4'

PORTFOLIO = q2.PORTFOLIO + [dict(q2._BC, lam=4.0, cache=True)]


def solve(graph, k, problem=3, seed=0, budget_s=None, warm=None, portfolio=None, **opts):
    if warm is None:
        warm = q2.archive_candidates(graph, k, problems=(3, 2, 1), smaller_k=(3, 2))
    return q2.solve(graph, k, problem=problem, seed=seed, budget_s=budget_s, warm=warm,
                    portfolio=PORTFOLIO if portfolio is None else portfolio, **opts)
