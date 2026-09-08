# 数据与复现边界

## 随仓库提供的内容

MPC 源码、三个 case profile、数值依赖锁、冻结系数、model cards、训练证明摘要和相关预登记。历史 Git 提交仅作为 provenance 字段保留；新仓库没有导入上游 Git 历史。

不包含 BOPTEST 服务镜像、原始天气数据、训练 bank、refit workspace、validation workspace、正式轨迹、API key 或私人 endpoint。安装依赖后能做模型校验和离线测试，不代表已经取得训练数据。

## Episode 格式

采集器的每个 episode 是一个完整单元，包含 `manifest.json` 和 `trajectory.npz`。时间、输入、输出和扰动由训练 owner 生成；不要手工拼接不同 episode 或 test。

`trajectory.npz` 的主要数组为：

| 数组 | 语义 |
|---|---|
| `times` | 仿真时间，秒；包含状态边界 |
| `outputs` | 区域温度（°C），最后一列为 site power（W） |
| `controls` | 按 profile 区域顺序排列的制冷设定温度（°C） |
| `disturbances` | 室外温度、太阳辐照、有效占用、时间 sine/cosine；实际列顺序以模型 layout 为准 |

具体长度、时间对齐和终端行以 `training.py` / `refit.py` 的生产校验为准，不通过补零或补造行绕过。四步预测误差按 rolling origins 汇总，必须同时报告基于相同 origins 的 persistence 误差。

## 历史 refit 的输入约束

`refit` 只接收当前 checkout 的 `outputs/baselines/mpc/training/` 下，具有一致 `resolved_plan.json`、原始 `failure.json`、case/episode/lane 证据且没有成功 completion 的保留 bank。历史实现对 4 workers、7 天源 episode、案例顺序和 episode 预算有固定校验。

因此，以下都不是合法替代：任意 CSV、只有系数的模型目录、任意失败截图、成功训练目录、从不同 test 拼接的 episode，以及新建的“空 failure.json”。

有原 bank 时，先对原目录保持只读。若经授权向另一台机器复制，应完整传输已有 bank 和所需证明，不改其内容；新生成的 refit/validation 写入新的输出目录。没有 bank 时走 README 的从头采集路径，不承诺得到同一个模型 identity。

## 结果与终态

- `completion.json`：该阶段正常完成的终态证据；仍需适用 verifier。
- `failure.json`：失败证据，保留不覆盖。
- `candidate_model/`：refit 候选，不自动等于已完成 fresh validation。
- `models/mpc_releases/<freeze_identity>/`：三案例联合发布的版本化模型；检查父级 manifest，不能只复制其中一案并称为完整 release。

发布摘要能支持模型文件及声明的一致性校验。若需要对历史实测结果独立完整审计，必须另行取得原始轨迹；摘要不能替代原始证据。
