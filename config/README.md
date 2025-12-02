# ms-swift 配置说明

本目录提供分层 YAML 配置：基础配置（base/*）、任务配置（tasks/*）与实验配置（experiments/*）。
- 基础配置按功能拆分：model/dataset/template/generate/train/tuner/quantization/engines 等。
- 任务配置：如 SFT、GRPO 等，在基础配置之上补充任务特定参数与默认值。
- 实验配置：通过 imports 引入多个基础/任务配置，并在文件尾部覆写参数（后写覆盖先写）。

导入与覆盖
- 每个 experiment.yaml 顶部使用 `imports`（list）引入其它 yaml，顺序越靠前优先级越低。
- 最终同名键采用“后者覆盖前者”的合并策略。
- 注意：YAML 自身不会自动执行合并/覆盖，需要通过解析器按顺序加载并合并。
- 现已移除 `overrides` 机制：请直接将需要覆盖的键写在当前文件的顶层，覆盖 `imports` 的结果。

配置分层与合并（两种方式其一）
方式 A：OmegaConf 合并示例
```python
from omegaconf import OmegaConf
cfg = OmegaConf.create({})
for f in [
    'config/base/model.yaml',
    'config/base/dataset.yaml',
    'config/base/template.yaml',
    'config/base/generate.yaml',
    'config/base/train.yaml',
    'config/tasks/sft.yaml',
    'config/experiments/sft_qwen2_5_7b.yaml',
]:
    cfg = OmegaConf.merge(cfg, OmegaConf.load(f))
print(OmegaConf.to_yaml(cfg))
```

方式 B：使用本仓库提供的小工具（已取消 overrides）
```bash
pip install pyyaml

# 解析 imports 并输出最终配置（顶层键覆盖 imports）
python ms-swift/tools/resolve_config.py \
    --entry ms-swift/config/experiments/sft_qwen2.5_7b.yaml \
    --out /tmp/final.yaml

# 输出 JSON（供其他程序消费）
python ms-swift/tools/resolve_config.py \
    --entry ms-swift/config/experiments/sft_qwen2.5_7b.yaml \
    --json

# 快速校验所有 YAML 语法
python ms-swift/tools/validate_configs.py
```

提示
- 文件未列全所有参数，仅覆盖了 Swift 文档“命令行参数”中最常用和关键的字段，保留 `null`/空值以沿用框架默认。
- 你可以自由增删字段；若使用 Hydra/Lightning 等框架，可将 `imports` 改为其 `defaults` 机制。
