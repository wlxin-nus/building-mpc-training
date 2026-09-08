# Building MPC Training

**用于建筑制冷控制的分层 MPC：数据采集、系统辨识、离线 refit、闭环验证与模型冻结。**

这是从 H3C 研究代码中独立提取的个人研究仓库。保留原有 MPC 数值方法与三个案例配置，去除了 LLM Agent、Prompt、Provider 调用、DRL 和论文工程。MPC 不需要 GPU，也不需要 Baseten/OpenAI/DeepSeek 密钥；**LLM 调用数、tokens 和 LLM 费用均为 0**。真实采集与闭环验证仍需要 BOPTEST 服务和相应计算资源。

> 本次拆分不是重新训练或调参。附带模型是既有冻结模型，不应称为新仓库重新训练的结果；原始训练 episode 和九条正式评估轨迹没有上传。

## 目录导航

- [快速开始](#快速开始)：安装、离线检查与 dry plan
- [训练与验证流程](#训练与验证流程)：从头训练和历史 refit 两条路径
- [方法与数据协议](#方法与数据协议)：模型、时间窗口、样本划分
- [冻结模型](#冻结模型)：三案例模型身份与真实验证状态
- [工程结构](#工程结构)、[常见问题](#常见问题)、[开发检查](#开发检查)
- 详细说明：[数据格式](docs/data_contract.md) · [来源与拆分边界](docs/extraction.md) · [原始预登记](docs/hierarchical_mpc_preregistration.md)

## 快速开始

### 1. 安装

推荐 Python **3.12**，支持范围声明为 3.11–3.13；本次离线测试使用 Windows / Python 3.12.2。已安装 Git 和 `uv` 后：

```powershell
git clone https://github.com/wlxin-nus/building-mpc-training.git
cd building-mpc-training
uv sync --locked --extra dev
uv run --no-sync building-mpc --help
```

仓库采用 **完整 checkout + editable install** 工作流：配置、模型及 Git 身份与源码一起使用，不支持只搬运一个 wheel。不要把它与完整 H3C 安装进同一个新环境，因为保留的 `h3c` / `h3c_baselines` Python 命名空间会重叠。

在原论文工作站上，继续复用 canonical 解释器，**不运行上述 `uv sync`，不创建或修改环境**：

```powershell
$python = 'D:\NUS\Paper\01-Heriachical Control\H3C_CAOL_Final_Worktree\.venv\Scripts\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& $python -m h3c_baselines.cli mpc --help
```

后文的 `building-mpc <命令>` 等价于 `python -m h3c_baselines.cli mpc <命令>`。

### 2. 离线检查附带模型

```powershell
$release = 'b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c'
uv run --no-sync building-mpc verify-frozen-suite --mpc-release $release
uv run --no-sync building-mpc verify-model --case SZ_Air --mpc-release $release
```

成功应返回 `valid: true`。发布状态 `METHOD-DEGRADED` 不等于文件损坏，具体含义见下文。此检查验证模型及发布证明的一致性；没有原始轨迹时，不代表重新完成历史物理回放审计。

### 3. 查看训练计划：不联网、不创建 test

```powershell
uv run --no-sync building-mpc train --case all --workers 4 --max-fit-episodes 64
```

返回 `execution: false`，显示案例顺序、episode 预算、warm-up 和时限。**只有显式加入 `--execute` 才执行对应操作**；其中 `refit` 的执行仍是纯离线。

## 训练与验证流程

### 路径 A：重新采集数据并训练

这是新的实验，不会精确重现历史数据或模型 identity。执行前确认 BOPTEST 已启动且可从当前进程访问、案例及版本正确、仓库干净并已提交、磁盘至少剩余 1 GiB。这个空间下限是启动门禁，不是整个训练的容量保证。

```powershell
# 示例地址；替换为你实际部署的、无凭据的 BOPTEST origin。
$env:BOPTEST_URL = 'http://127.0.0.1:5000'
uv run --no-sync building-mpc train --case all --workers 4 --max-fit-episodes 64 --execute
```

- 原始采集器按 **SZ Air → MZ Hydro → MZ Air** 顺序处理案例，同一案例使用 4 条独立 lane；不是同时启动三案例各 4 条。
- checkpoint 为 8、16、32、64 条 fit episode，另有 holdout、参考及闭环验证 episode；64 不是总 episode 数。
- 每个 lane 选择自己的 test，同一 lane 的各 episode 重新 initialize 并执行 7 天内部 warm-up；任务结束释放 test。fresh closed-loop validation 则为每案另选 fresh test。
- 6 小时是原始采集任务的总 wall-clock 预算，不是模型保证收敛时间。
- 成功输出 `outputs/baselines/mpc/training/<run-id>/completion.json`，模型写入 `models/mpc/<case>/`。已有模型目标会拒绝覆盖。
- 无合格 checkpoint、基础设施故障或证据门禁失败时保留 `failure.json`，不要删除失败目录、伪造 completion 或因性能差反复补跑。

原始 CLI 可以解析较小 worker 数及 episode 上限，但历史完整协议和 refit 身份校验按 **4 workers** 设计；复现请使用上面的默认完整配置。

### 路径 B：从已有历史失败数据 bank 离线 refit

这是附带新版模型的实际来源路径。`refit` 是**保留失败训练 bank 的恢复入口**，不是任意 CSV/NPZ 拟合器，也不能把路径 A 的成功目录直接传给它。

原始 bank 默认不随仓库分发。如果你有它的经授权副本，应完整保留其目录和证据，将其放在本仓库 `outputs/baselines/mpc/training/<source-run>/` 下；不修改原 bank，不自行补造 failure marker。

```powershell
$source = 'outputs/baselines/mpc/training/<source-run>'
uv run --no-sync building-mpc refit --source-run $source --align-training-windows
uv run --no-sync building-mpc refit --source-run $source --align-training-windows --execute
```

命令返回新的 `run_dir`。以下命令中的 `<refit-run>` 和 `<validation-run>` 必须换成实际返回的目录，不能直接复制占位符运行：

```powershell
$refit = 'outputs/baselines/mpc/refit/<refit-run>'
uv run --no-sync building-mpc verify-refit $refit
uv run --no-sync building-mpc validate --refit-workspace $refit --publication versioned

# 真实闭环验证：最多三个案例同时运行，使用各自 fresh test。
uv run --no-sync building-mpc validate --refit-workspace $refit --publication versioned --execute

uv run --no-sync building-mpc verify-validation 'outputs/baselines/mpc/validation/<validation-run>'
```

验证使用相同候选系数、原始四步 horizon、共同权重和缩放，不进行新的超参数搜索。`--publication versioned` 将完整三案例 suite 写入 `models/mpc_releases/<freeze_identity>/`，拒绝覆盖已有身份。验证和发布的自动衔接继承原预登记：性能降级可以被明确分类接受，执行完整性失败不可发布。

| 验证结果 | 含义与处理 |
|---|---|
| `BASELINE-READY` | 证据有效、fallback=0、occupied peak `|PMV|≤0.70` |
| `METHOD-DEGRADED` | 证据有效，但有 fallback 或超过该舒适边界；保留不利结果并按已登记路径发布，不伪装为 clean pass |
| 身份、轨迹、非有限值、secret 或生命周期失败 | 停止；不可发布、不覆盖原目录 |

`freeze-adverse` 是显式降级发布入口，不是放宽完整性检查的开关。原始默认路径 `models/mpc` 仍保留供兼容；**新版研究工作使用 versioned 发布**。

## 方法与数据协议

MPC 在每次控制时求解优化问题；这里的“训练”主要是**预测模型辨识**，不是强化学习，也不是让优化器持续学习目标权重。

| 组成 | 当前实现 |
|---|---|
| 预测模型 | 多输出 vector ARX：各区域空气温度 + 全楼功率 |
| 历史 | 当前输出及前 4 个完成记录；控制量使用候选输入及前 4 个完成记录 |
| 拟合 | 标准化 ridge regression；alpha 候选 `1e-6, 1e-4, 1e-2, 1, 100` |
| 预测/控制时域 | 4 × 900 s = 1 h |
| 控制结构 | 每小时楼级协调器 + 每 15 分钟区域 MPC；一次反馈协调 |
| 舒适度 | 共享 PMV owner、内部线性近似、校准 episode 的残差裕量 |
| 优化器 | OSQP 1.1.3；原容差、迭代预算、polishing 和 fallback 逻辑不变 |
| 跨案例原则 | 相同算法、目标权重、缩放和规则；点位、占用与物理尺度由 profile 声明 |

### 三类窗口不能混为一谈

以下 day 是配置中的 BOPTEST 仿真日索引，区间右端不含；不是电脑日期。

| 案例 | 新版模型 fit/holdout/calibration 日历窗口 | fresh validation | 历史正式评估窗口 |
|---|---|---|---|
| SZ Air | `[196, 203)`，7 天 | day 196 起 167 h | `[203, 210)`，168 h |
| MZ Hydro | `[213, 218)`，5 天 | day 213 起 167 h | `[220, 225)`，120 h |
| MZ Air | `[192, 199)`，7 天 | day 192 起 167 h | `[199, 206)`，168 h |

- 原始采集 bank 的各 episode 为 7 天；`--align-training-windows` 在 refit 时为 Hydro 选取 5 天前缀，其他两案保留 7 天。
- 同一开发日历窗口有多条不同激励 episode，不是“只训练一条 7 天轨迹”。
- holdout 不进入系数拟合，但用于 ridge 选择与 persistence 比较，所以不是完全未参与选择的最终测试集。
- PMV 校准使用单独角色的 episode；不能把残差分位数写成全时段舒适保证。
- fresh validation 为 668 × 900 s = 167 h，末 1 h 留给 forecast。fresh test ID 表示新物理实例，不意味着新的天气日期。

## 冻结模型

附带 release：

```text
b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c
```

完整 identity 与 provenance 在各 [model card](models/mpc_releases/b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c/) 中；缩写仅方便阅读。

| 案例 | 模型 identity（缩写） | fresh validation fallback | occupied peak `|PMV|` | 分类 |
|---|---|---:|---:|---|
| SZ Air | `62b361…7791fd1` | 1 | 0.61 | `METHOD-DEGRADED` |
| MZ Hydro | `29c104…a85413` | 0 | 0.63 | `BASELINE-READY` |
| MZ Air | `4e38b2…991529` | 2 | 0.90 | `METHOD-DEGRADED` |

这是固定配置、有限开发日历窗口下的代表性 MPC 基线，**不声称已充分整定、全局最优或优于 eRBC**。低预测误差也不单独证明闭环控制最优。完整三轮正式评估和 H3C 对比保留在原研究工程；本仓库聚焦训练，不包含正式 campaign launcher 或其原始证据。

## 工程结构

```text
configs/                       三案例 profile、MPC 配置、参考 RBC 程序
src/h3c_baselines/mpc/
  vector_arx.py                特征、标准化、拟合、预测及模型序列化
  training.py                  激励采集、episode 划分、checkpoint 选择
  refit.py                     保留 bank 的离线重建、校准与候选门禁
  forecast.py                  公共观测与四步 forecast 对齐
  optimizer.py                 楼级/区域级 MPC 与 fallback
  validation.py               fresh test 闭环验证和证据核对
  registry.py                 三案例原子、版本化模型发布
src/h3c_baselines/cli.py        MPC 专用命令入口
src/h3c/                       必要的物理、PMV、占用及参考程序公共模块
models/mpc_releases/            不可覆盖的既有冻结模型
tests/                         不出站的单元、fake-physical 和拆分回归检查
docs/                          数据协议、来源说明、原始预登记
provenance.json                上游提交与逐文件来源
```

`configs/graphs/` 只用于保持原 profile 合同完整，不启用因果控制，也不导入 Agent。原命名空间保留是为了不重命名数值 owner，不表示仍依赖完整 H3C 工程。

## 开发检查

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src
uv lock --check
```

测试用 fake physical client，不创建真实 BOPTEST test。测试、静态检查与 frozen-suite 验证通过，并不等同于新仓库已经完成一次真实再训练。本次提取的具体检查记录见 [来源与验证](docs/extraction.md)。

## 常见问题

**没有 BOPTEST 可以做什么？** 可以读模型、校验冻结套件、运行离线测试和 dry train plan。refit 还需合规的本地原始 bank；不附带合成数据冒充真实数据。

**为什么仓库没有训练数据？** 原始轨迹、失败证据、临时日志和服务地址不进入新 repo，既保护隐私，也避免把研究归档误当软件发行内容。格式与获取边界见 [数据协议](docs/data_contract.md)。

**连接失败怎么办？** 确保真实启动进程所在网络上下文能连接 `BOPTEST_URL`。CLI 先做 TCP 检查，失败不创建 test。不要把 socket permission denied 当成 MPC 性能或模型问题。

**dirty worktree / existing lock 为什么停止？** 真实执行需要可追踪的已提交源码，且不能共享运行写入者。先检查 owner；不要为了通过门禁删除未知锁或覆盖旧结果。

**可以调整权重、缩放或逐案例优化吗？** 这次独立化没有进行这些操作。以后改变方法必须生成新的研究身份并说明验证协议；不能让旧 freeze 或历史结果替变化后的实现背书。

## 许可与隐私

源码延续上游 [MIT License](LICENSE) 和原版权声明；GitHub 仓库设置为 private。BOPTEST 软件、天气及案例数据遵循各自许可。凭据只能通过本地环境提供，`.env` 不提交；本项目没有 LLM 请求入口。不要把原论文仓库或原始实验目录整体复制进这里。
