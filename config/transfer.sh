#!/usr/bin/env bash
set -euo pipefail

# transfer：
# 用法：
#   ./transfer.sh experiments/sft_qwen2.5_7b.yaml
# 效果：
#   在 config/out/ 下生成 {experiment}_{YYYYMMDD_HHMMSS}.yml 的合并结果。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="${SCRIPT_DIR}/../tools/resolve_config.py"
OUT_DIR="${SCRIPT_DIR}/out"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <experiment_yaml_path>" >&2
  exit 1
fi

ENTRY="$1"
# 若未提供绝对路径，则相对 config 目录解析
if [[ "${ENTRY}" != /* ]]; then
  ENTRY="${SCRIPT_DIR}/${ENTRY}"
fi

if [[ ! -f "${ENTRY}" ]]; then
  echo "错误：找不到 experiment 文件: ${ENTRY}" >&2
  exit 1
fi

# 依赖检查：pyyaml
if ! python -c 'import yaml' >/dev/null 2>&1; then
  echo "检测到缺少 pyyaml，正在尝试安装 (--user)..." >&2
  python -m pip install --user pyyaml >/dev/null
fi

mkdir -p "${OUT_DIR}"
base_name="$(basename -- "${ENTRY}")"
stem="${base_name%.*}"
ts="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/${stem}_${ts}.yml"

python "${RESOLVER}" --entry "${ENTRY}" --out "${OUT_FILE}"
echo "已生成: ${OUT_FILE}"