. /home/zqiu/anaconda3/etc/profile.d/conda.sh
module load cuda/12.9
conda activate verl

export WANDB_PROJECT="verl_grpo_example_gsm8k"
export WANDB_NAME="qwen2.5_3b_grpo_lora"

# # Resolve repo root and source wandb_api.sh from there
# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

# [ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"

export VLLM_TORCH_COMPILE_LEVEL=0

bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh