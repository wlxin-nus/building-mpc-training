# 独立仓库来源与验证

## 来源

- 提取日期：2026-09-08。
- 上游源码提交：`32524f3b3fd3861a9323226612c776b27f4b8a85`。
- 附带模型 release：`b89a2138daa2e32867768125ee84cf607927ef615c5f42065487c11b5d9e635c`。
- 新仓库从独立初始提交开始，不导入上游论文、Agent 或实验 Git 历史。

## 保留与改变

七个 MPC 算法/训练/发布模块、case/MPC 配置、PMV 与占用计算保持上游内容。`provenance.json` 记录逐文件来源，文本只允许 Git 常规换行归一化。冻结 release 文件不转换换行，以保留内部证据 checksum。

独立化仅改变工程边界：

1. CLI 只保留 MPC 命令；移除 DRL、正式 suite、全项目报告和 LLM 命令。
2. HTTP 模块保留上游 BOPTEST 类与其必要辅助函数，删除 LLM client。JSON 序列化辅助函数的历史名称不代表存在模型请求入口。
3. 参考 RBC 的解释器只保留 load/validate/execute 及必要定义，不包含 Agent patch 应用和候选推导。
4. secret 检查不再依赖 LLM runtime 配置，仍检查已配置的已知密钥值，且不打印值。
5. 启动预检从论文工作站的硬编码解释器路径改为当前 checkout、干净 Git、锁、磁盘、数值依赖及同进程 TCP 检查。原工作站仍使用 canonical 环境。
6. 依赖移除 Torch、DRL、Agent Framework 等非 MPC 包；实际数值库版本按现有 canonical 环境固定。

未修改 ARX 系数、训练数据、ridge 选择、缩放、权重、约束、horizon、OSQP 设置、fallback、profile、PMV 或 Reward。没有进行真实训练、额外诊断或 BOPTEST 调用。

`configs/graphs/` 是 profile 合同要求的兼容文件；运行 MPC 不消费因果图。保留它们避免为拆分重写 profile 验证。

## 验证范围

交付检查包括现有 MPC 离线测试、拆分回归、Ruff、format、strict mypy、依赖锁、无副作用 dry plan、三案例冻结模型一致性、来源一致性及发布前 secret/文件清单检查。没有在新仓库上重新采集真实轨迹；不能把离线检查写成九条正式结果的重新认证。

2026-09-08 本地核验结果：

- Python 3.12.2 canonical 解释器；NumPy 2.2.6、SciPy 1.15.3、OSQP 1.1.3、pythermalcomfort 3.9.8。
- **106 tests passed**；禁止测试出站的 fixture 生效。
- Ruff check、36 文件 format check、30 source files strict mypy 均通过。
- `uv lock --check` 和 wheel 构建通过；没有修改 canonical 环境。
- 附带三案例 frozen-suite 全部 `case_checks=true`，`valid=true`，原 `METHOD-DEGRADED` 分类保留。
- 57 份保留文件与上游一致；两处抽取模块的全部保留函数/类 AST 与上游一致。
- dry training plan 不访问 BOPTEST，不产生 runtime 输出。

## 历史研究记录

- [原始分层 MPC 预登记](hierarchical_mpc_preregistration.md)
- [公共观测 refit 预登记](mpc_common_observation_refit_preregistration_20260905.md)
- [第四步 forecast 可用性预登记](mpc_t_plus_4_forecast_availability_preregistration_20260905.md)
- [新版三轮评估与版本化发布预登记](mpc_common_observation_formal_repeats_preregistration_20260907.md)

这些文档描述上游研究的真实时间顺序与批准边界。它们包含上游工程路径或命令，不是新仓库的当前操作手册；安装和命令以本仓库 README 为准。
