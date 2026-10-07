# DAGShift · 神经网络计算图多核划分与调度

[![Checks](https://github.com/Ho-boop/dagshift/actions/workflows/ci.yml/badge.svg)](https://github.com/Ho-boop/dagshift/actions/workflows/ci.yml)

DAGShift 是一个 Python 计算图划分与调度求解器项目：输入 Op–Tensor DAG 和核心数量，输出计算操作的子图归属，以及每个核心的子图执行顺序。项目结合结构化构造、有限定向改进和确定性预算控制，关注方案可行性、计算时延与数据搬运开销。

**个人独立 AI Coding 项目。** 开发中使用 Claude Code、Codex 多会话协作，由作者负责需求与约束梳理、模块分工、接口与文件边界、实验对照、代码验收和最终交付。[AI 工作流](AI_WORKFLOW.md)描述如何组织这套开发过程。

本仓库包含正式交付路径的求解器源码，以及一个无需竞赛附件的合成图演示。竞赛原题、官方代码、官方数据、衍生加速评估器、论文和会话日志不随仓库发布。完整求解使用用户自行提供的外部评估器；演示输出的计算负载是代理量。

## Quick start

Python 3.14；可移植演示仅使用标准库，无第三方依赖。

```powershell
git clone https://github.com/Ho-boop/dagshift.git DAGShift
cd DAGShift
$env:PYTHONUTF8 = '1'
python -m dagshift.demo --cores 3 --out demo-output
python -m unittest discover -s tests -v
```

打开 `demo-output/report.html` 可以查看每个核心的计算工作量。运行同时输出标准方案 `plan.json` 与独立报告 `report.json`。输入样例由本项目手工生成，位于 [`examples/synthetic.json`](examples/synthetic.json)。

```powershell
python -m dagshift.demo --graph examples/synthetic.json --cores 2 --out demo-output/two-core
```

演示采用弱连通分量按计算工作量贪心装箱，不切分分量。它复用正式项目的图表示、方案编码和结构验证，但不运行完整优化算法，也不模拟 DDR、Spill、Pipe 排队或 L2 Cache。报告中的 `largest_core_compute_work` **不是官方 Makespan**。

## What is implemented

| 模块 | 实现 |
|---|---|
| 图表示 | 收缩 COPY 路径形成计算 DAG；缓存拓扑序、关键路径、分量、Pipe 工作量及张量共享关系 |
| 问题一 | 分量装箱、层带和批处理构造、候选预排序、受限局部改进 |
| 问题二/三 | 列表调度、多个结构起点、块与关键链改进、固定分核的核内重排、深图追加候选 |
| 可复现执行 | 固定种子、基于评估工作量的虚拟时钟、真实时间保护、同次运行 k−1 方案复用 |
| 方案验证 | 格式、覆盖、子图 DAG、同核顺序及问题一 Task 顺序环检查；完整执行可行性交给外部评估器 |
| 交付追溯 | 输入和方案哈希、代码版本、配置、预算、耗时、回退与 cap 状态独立记录 |
| 可移植演示 | 合成图输入、方案生成、独立 JSON 报告、HTML 可视化及 CI |

## Input and output

图输入顶层为 `ops`、`tensors`、`edges` 三个数组，操作与张量使用不相交的整数 ID。

```json
{
  "ops": [
    {"id": 0, "op": "MATMUL", "pipe": "PIPE_M", "cycles": 80},
    {"id": 1, "op": "ADD", "pipe": "PIPE_V", "cycles": 60}
  ],
  "tensors": [{"id": 1000, "pos": "UB", "size": 64}],
  "edges": [{"source": 0, "target": 1000}, {"source": 1000, "target": 1}]
}
```

操作包括 `id`、`op`、`pipe`、`cycles`；张量包括 `id`、`pos`、`size`；边包括 `source`、`target`，直接操作边可含 `data_size`。`COPY_IN`/`COPY_OUT` 保留在原图，但不分配子图。演示使用上述最小结构；完整实例还需满足外部评估器的参数与内存语义约束。

方案顶层严格为两个字段，示例：

```json
{
  "node_to_subgraph": {"0": 0, "1": 0},
  "core_schedules": [[0], []]
}
```

`node_to_subgraph` 恰好覆盖所有非 COPY 计算操作。每个子图在 `core_schedules` 中恰好出现一次；外层数组长度为核数，可有空核。日志、报告和运行元数据均放在方案文件之外。

## Run the complete solvers with your own evaluator

### Option A: original official package

将你合法取得的官方附件保留在仓库外。该目录应具有原始 `code/` 与 `data/config.txt`；本项目适配层直接调用原始 Python 评估函数，不复制或修改它们。

```powershell
$env:DAGSHIFT_OFFICIAL_PACKAGE = 'E:\private-assets\official-package'
python -m solvers.deliver --problem 1 --graph examples/synthetic.json --k 2 --budget 1 --out demo-output/q1
python -m solvers.deliver --problem 2 --graph examples/synthetic.json --k 2 --budget 1 --append deep --out demo-output/q2
```

原始评估器可能较慢；其运行速度和虚拟时钟触发情况与历史优化评估器不同，不能直接复用下方历史耗时。`--append deep` 仅适用于问题二/三的 `virtual` 模式。

### Option B: separately supplied accelerated evaluator

如果你另有兼容的 `fast_eval` 包，可设 `DAGSHIFT_EVALUATOR_ROOT` 为其父目录；这一选项优先于 Option A。该包必须提供 `load_config`、`evaluate`、`singlecore`、`set_quiet`，以及求解器使用的 `schedule_step1` 和问题二/三子模块接口。它是外部依赖，不受本仓库 MIT 许可覆盖。

```powershell
$env:DAGSHIFT_EVALUATOR_ROOT = 'E:\private-assets\evaluator-workbench'
$env:DAGSHIFT_CONFIG = 'E:\private-assets\official-package\data\config.txt'
python -m solvers.deliver --problem 3 --graph '<your-graph.json>' --k all --append deep --out demo-output/q3
```

可用 `DAGSHIFT_DATA_DIR` 指定批量输入目录。`DAGSHIFT_CONFIG` 指定配置；否则读取 `DAGSHIFT_OFFICIAL_PACKAGE/data/config.txt`。`--config` 仅检查一致性。每个 `(图, 问题)` 使用一个进程；同进程不可多线程并发调用虚拟时钟求解器。

| 运行参数 | 默认值与范围 |
|---|---|
| `--k` | `all`，即 1～5；也可给单核数或逗号列表 |
| `--clock` | `virtual`；`wall` 作真实时间截止的对照 |
| `--budget` | 120，k=2～5 共享的虚拟预算；不是端到端墙钟承诺 |
| `--seed` | 0 |
| `--cap-factor` | 6；真实保护触发后记录 `capped=True` |
| `--append deep` | 问题二/三启用交付 v3 追加候选；默认关闭，对应 v2 |

同输入、固定版本、种子和预算下，且真实保护未触发时，虚拟计费降低了机器负载对搜索停止位置的影响。跨 Python 版本的一致性未作承诺。求解入口将历史最优库、旧单核表和磁盘评估缓存隔离到本次运行的临时目录，不用历史答案查表。

## Historical measurements

以下是**2026-09 的本地交付 v3 历史实验摘要**，不是本次公共仓库重跑成绩。三问均为 100 例、k=2～5，无历史初解、固定种子，使用当时的优化评估器、i9-14900HX（32 线程、64 GB）、Python 3.14.2；案例计时包含四种核数端到端求解。不同实验分块并发，最多 16 个进程；耗时受负载影响。

| 场景 | k=2 平均加速比 | k=3 | k=4 | k=5 | 案例最大墙钟 |
|---|---:|---:|---:|---:|---:|
| 问题一 | 2.0325 | 2.8887 | 3.6302 | 4.2364 | 356.463 s |
| 问题二 | 2.3015 | 3.2830 | 4.1186 | 4.8129 | 379.221 s |
| 问题三 | 2.3002 | 3.3077 | 4.1531 | 4.9120 | 392.279 s |

加速比在这张表中均定义为固定无 Cache 单核基准 / 多核 Makespan 的逐例均值；问题三这一列不表示 L2 本身带来的加速。三问各 100/100 案例在项目内部 600 秒口径内完成；600 秒是工程验收标准。历史优化评估器在 100 例 × 13 类方案上与官方结果对象一致；后续精简交付包的官方 CLI **20/20 抽查**一致。未声称 1500 份交付方案全量官方 CLI 复评。

摘要数据和口径见 [`docs/HISTORICAL_RESULTS.md`](docs/HISTORICAL_RESULTS.md) 与 [`docs/historical_summary.csv`](docs/historical_summary.csv)。原始竞赛数据和论文未发布，读者需自行取得对应输入才能复现这些历史测量。公共版改变了外部依赖加载，不改变优化算法；本次只运行小合成图验证。

## Repository map

```text
dagshift/             可移植演示和外部评估器适配
common/               图表示、方案编码、结构预筛、评估摘要与缓存
solvers/q1/           问题一求解器
solvers/q2/           问题二主算法及正式使用的 dev 模块
solvers/q3/           问题三入口
solvers/deliver/      隔离求解、哈希记录与外部 CLI 核对
solvers/vclock.py     确定性工作量计费与真实保护
examples/            手工合成输入
tests/               不依赖官方资产的回归检查
docs/                历史摘要、源码来源哈希与发布验证
```

MIT 许可仅适用于本仓库的自写代码与文档。外部竞赛材料及衍生评估器按各自的权利与许可处理。
