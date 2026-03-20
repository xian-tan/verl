#!/bin/bash
###############################################################################
# DAAS 实验 — 批量运行所有实验（Math 领域）
#
# 自动依次运行 4 种 shaping 策略，也可以传入额外参数进行 alpha/beta 消融实验
#
# 使用方法:
#   bash run_all_daas_math.sh                     # 运行全部4组实验（默认参数）
#   bash run_all_daas_math.sh --ablation-alpha     # alpha 消融实验
#   bash run_all_daas_math.sh --ablation-beta      # beta 消融实验
###############################################################################

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE=${1:-"default"}

if [ "${MODE}" == "--ablation-alpha" ]; then
    echo "===== Alpha 消融实验 ====="
    for alpha in 0.5 1.0 2.0 3.0; do
        echo ">>> Running hard_focused with alpha=${alpha}"
        bash "${SCRIPT_DIR}/run_daas_math.sh" hard_focused ${alpha}
    done
elif [ "${MODE}" == "--ablation-beta" ]; then
    echo "===== Beta 消融实验 ====="
    for beta in -2.0 -1.0 0.0 1.0 2.0; do
        echo ">>> Running temperature with beta=${beta}"
        bash "${SCRIPT_DIR}/run_daas_math.sh" temperature 1.0 ${beta}
    done
else
    echo "===== 全部4组实验 ====="
    echo ">>> 1/4: Baseline GRPO"
    bash "${SCRIPT_DIR}/run_daas_math.sh" baseline

    echo ">>> 2/4: Easy-focused (alpha=1.0)"
    bash "${SCRIPT_DIR}/run_daas_math.sh" easy_focused 1.0

    echo ">>> 3/4: Hard-focused (alpha=1.0)"
    bash "${SCRIPT_DIR}/run_daas_math.sh" hard_focused 1.0

    echo ">>> 4/4: Temperature (beta=1.0)"
    bash "${SCRIPT_DIR}/run_daas_math.sh" temperature 1.0 1.0
fi

echo "===== 全部实验完成 ====="
