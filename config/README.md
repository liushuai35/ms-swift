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


CUDA_VISIBLE_DEVICES: 控制使用哪些GPU卡。默认使用所有卡。

ASCEND_RT_VISIBLE_DEVICES: 控制使用哪些NPU卡（只对ASCEND卡生效）。默认使用所有卡。

MODELSCOPE_CACHE: 控制缓存路径。（多机训练时建议设置该值，以确保不同节点使用相同的数据集缓存）

PYTORCH_CUDA_ALLOC_CONF: 推荐设置为'expandable_segments:True'，这将减少GPU内存碎片，具体请参考torch文档。

NPROC_PER_NODE: torchrun中--nproc_per_node的参数透传。默认为1。若设置了NPROC_PER_NODE或者NNODES环境变量，则使用torchrun启动训练或推理。

MASTER_PORT: torchrun中--master_port的参数透传。默认为29500。

MASTER_ADDR: torchrun中--master_addr的参数透传。

NNODES: torchrun中--nnodes的参数透传。

NODE_RANK: torchrun中--node_rank的参数透传。

LOG_LEVEL: 日志的level，默认为'INFO'，你可以设置为'WARNING', 'ERROR'等。

SWIFT_DEBUG: 在engine.infer(...)时，若设置为'1'，PtEngine将会打印input_ids和generate_ids的内容方便进行调试与对齐。

VLLM_USE_V1: 用于切换vLLM使用V0/V1版本。

SWIFT_TIMEOUT: (ms-swift>=3.10) 若多模态数据集中存在图像URL，该参数用于控制获取图片的timeout，默认为20s。

ROOT_IMAGE_DIR: (ms-swift>=3.8) 图像（多模态）资源的根目录。通过设置该参数，可以在数据集中使用相对于 ROOT_IMAGE_DIR 的相对路径。默认情况下，是相对于运行目录的相对路径。

SWIFT_SINGLE_DEVICE_MODE: (ms-swift>=3.10) 单设备模式，可选值为"0"(默认值)/"1"，在此模式下，每个进程只能看到一个设备