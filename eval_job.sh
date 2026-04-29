# export HF_HUB_OFFLINE=1 # Set Hugging Face Hub to offline mode to run in interactive sessions
export BATCH_SIZE=1

# TASK_GROUPS="mgsm-eu,xcsqa,open-subtitles-x-to-eng,open-subtitles-eng-to-x,doclevel-mt-x-to-eng,doclevel-mt-eng-to-x"
# TASK_GROUPS="mgsm-eu" # queued
# TASK_GROUPS="xcsqa" # queued
# TASK_GROUPS="doclevel-mt-x-to-eng,doclevel-mt-eng-to-x" # queued
# TASK_GROUPS="polymath" # queued
# TASK_GROUPS="global-piqa" # queued with qwen
TASK_GROUPS="polymath" # queued with qwen
# TASK_GROUPS="sib-200,mgsm-eu,xcsqa,doclevel-mt-x-to-eng,doclevel-mt-eng-to-x,polymath"
# TASK_GROUPS="belebele-eu-cf,sib-200,multiblimp,global-piqa,mgsm-eu,xcsqa,open-subtitles-x-to-eng,open-subtitles-eng-to-x,doclevel-mt-x-to-eng,doclevel-mt-eng-to-x,polymath"

args=(
  # --venv_path .venv
  # --download_only true
  # --slurm_template_var '{"PARTITION":"dev-g","TIME":"01:00:00","GPUS_PER_NODE":1}' # when small-g is busy
  # --dry_run
  # --eval_csv_path "multisynt-evaluations/multisynt_evals_cf.csv"
  # --eval_csv_path "multisynt-evaluations/multisynt_evals_cf_9b.csv"
  # --eval_csv_path "multisynt-evaluations/multisynt_evals_cf_others.csv"
  # --models "EleutherAI/pythia-14m"
  --models "Qwen/Qwen2.5-0.5B"
  # --models "MultiSynt/nemotron-cc-finnish-tower72b"
  # # --task_groups "belebele-eu-cf"
  --task_groups "${TASK_GROUPS}"
  # --task_groups "sib-200"
  # --task_groups "multiblimp"
  # --task_groups "global-piqa"
  # --task_groups "mgsm-eu"
  # --task_groups "xcsqa"
  # --task_groups "open-subtitles-x-to-eng"
  # --task_groups "open-subtitles-eng-to-x"
  # --task_groups "doclevel-mt-x-to-eng"
  # --task_groups "math"
  # --task_groups "doclevel-mt-eng-to-x"
  # --tasks "mgsm_native_cot_bn"
  # --tasks "belebele_fin_Latn_cf"
  --n_shot 0
  --limit 4
)

uv run oellm schedule-eval "${args[@]}"

# saves result to $EVAL_OUTPUT_DIR
# /leonardo_work/OELLM_prod2026/users/tko00000/oellm-evals/outputs
# /pfs/lustrep4/scratch/project_462000963/oellm-cli-shared-evals/tingwenk/
# if --local is set, to ./oellm-output 