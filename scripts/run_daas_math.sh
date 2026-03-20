#!/bin/bash
###############################################################################
# Difficulty-Aware Advantage Shaping (DAAS) 实验脚本 — Math 领域
#
# 使用方法:
#   bash run_daas_math.sh baseline       # 标准 GRPO（对照组）
#   bash run_daas_math.sh easy_focused   # Easy-focused shaping
#   bash run_daas_math.sh hard_focused   # Hard-focused shaping
#   bash run_daas_math.sh temperature    # Temperature shaping
#
# 训练集: MATH + SVAMP
# 评估集: GSM8K (easy), SVAMP (medium), MATH (hard)
###############################################################################

set -x

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWANLAB_API_KEY=yGRB9Af71SSb9z3GtnvDf

# ===================== 实验策略选择 =====================
STRATEGY=${1:-"baseline"}
ALPHA=${2:-1.0}      # easy/hard-focused 的 alpha 参数
BETA=${3:-1.0}       # temperature shaping 的 beta 参数

# 移除已使用的位置参数，剩余的 $@ 可用于传递额外的 hydra override
shift $(( $# < 3 ? $# : 3 ))

echo "============================================"
echo "  DAAS 实验: strategy=${STRATEGY}"
echo "  alpha=${ALPHA}, beta=${BETA}"
echo "============================================"

# 根据策略设置 adv_estimator
case ${STRATEGY} in
    "baseline")
        ADV_ESTIMATOR="grpo"
        EXP_NAME="daas_math_baseline_grpo"
        ;;
    "easy_focused")
        ADV_ESTIMATOR="grpo_easy_focused"
        EXP_NAME="daas_math_easy_focused_alpha${ALPHA}"
        ;;
    "hard_focused")
        ADV_ESTIMATOR="grpo_hard_focused"
        EXP_NAME="daas_math_hard_focused_alpha${ALPHA}"
        ;;
    "temperature")
        ADV_ESTIMATOR="grpo_temperature"
        EXP_NAME="daas_math_temperature_beta${BETA}"
        ;;
    *)
        echo "Unknown strategy: ${STRATEGY}"
        echo "Usage: bash run_daas_math.sh [baseline|easy_focused|hard_focused|temperature] [alpha] [beta]"
        exit 1
        ;;
esac

echo "ADV_ESTIMATOR=${ADV_ESTIMATOR}"
echo "EXP_NAME=${EXP_NAME}"

# ===================== 训练配置 =====================
# 注意: 请根据实际路径修改以下变量
MODEL_PATH=${MODEL_PATH:-"/models/Qwen2.5-3B-Instruct"}
N_GPUS=${N_GPUS:-8}
ROLLOUT_N=${ROLLOUT_N:-8}  # group size，用于动态计算 difficulty
gsm8k_train_path=../data/gsm8k/train.parquet
math_train_path=../data/math/train.parquet
math_dapo_train_path=../data/math_dapo/sample_train.parquet
gsm8k_test_path=../data/gsm8k/test.parquet
math_test_path=../data/math/test.parquet
math_dapo_test_path=../data/math_dapo/sample_test.parquet

# train_files="['$math_dapo_train_path']"
train_files="['$gsm8k_train_path', '$math_train_path', '$math_dapo_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path', '$math_dapo_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    +algorithm.adv_shaping_alpha=${ALPHA} \
    +algorithm.adv_shaping_beta=${BETA} \
    +algorithm.difficulty_thresholds='[0.4, 0.7]' \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.val_before_train=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='daas_experiments_mix' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.balance_batch=True \
    $@

echo "============================================"
echo "  实验完成: ${EXP_NAME}"
echo "============================================"
