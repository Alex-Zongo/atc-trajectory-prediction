#!/usr/bin/env bash
# The v2 training and evaluation plan, by stage. Each stage runs its jobs one after the
# other (one GPU); logs go next to the checkpoints in runs/v2/.
#
#   bash scripts/run_v2.sh data       # windows, word confidence, messages, data figures
#   bash scripts/run_v2.sh core       # main SBM + main CTP + history-only + per-step CTP
#   bash scripts/run_v2.sh baselines  # MDN, anchors, direct transcript (scratch + SBM init)
#   bash scripts/run_v2.sh ablations  # SBM factorizations, uniform flow time, no history
#   bash scripts/run_v2.sh seeds      # training seeds 1 and 2 for the main models
#   bash scripts/run_v2.sh channels   # messages + SBM for the other recognizer profiles
#   bash scripts/run_v2.sh eval       # every model on the test split + composition analysis
#   bash scripts/run_v2.sh diagnose   # H1-H5 and the SBM checks
#   bash scripts/run_v2.sh sweeps     # composition sweeps: K x S, sampler, CTP and SBM variants
#
# WANDB=1 bash scripts/run_v2.sh core   logs every run to W&B (project atc-v2, group = stage)
set -euo pipefail
cd "$(dirname "$0")/.."
R=runs/v2
mkdir -p "$R" results/v2
py() { uv run python "$@"; }
W=()
if [ -n "${WANDB:-}" ]; then W=(--wandb --wandb-group "${1:-}"); fi
traj() { py -m atc_forecast.training.traj_v2 "$@" ${W[@]+"${W[@]}"}; }
sbm() { py -m atc_forecast.training.sbm_v2 "$@" ${W[@]+"${W[@]}"}; }

case "${1:-}" in
data)
  py -m atc_forecast.data.windows_v2 --adsb "data/adsb/adsb_*.parquet" --out data/v2
  # word confidence is fitted to recognizer benchmarks (data/asr, internal); without them the
  # message build uses a parametric fallback of the same shape (channel/word_conf.py)
  if [ -d data/asr ]; then py -m atc_forecast.channel.word_conf --out data/v2/word_conf.json
  else echo "no data/asr: messages use the parametric word-confidence fallback"; fi
  py -m atc_forecast.data.messages --data data/v2 --profile jacktol
  py scripts/inspect_v2.py --data data/v2
  ;;
core)
  sbm  --factorization anyorder --out $R/sbm_jacktol_anyorder_s0.pt --seed 0   > $R/sbm_jacktol_anyorder_s0.txt
  traj --model ctp --output basis --condition atoms --out $R/ctp_basis_atoms_s0.pt --seed 0 > $R/ctp_basis_atoms_s0.txt
  traj --model ctp --output basis --condition none  --out $R/ctp_basis_none_s0.pt  --seed 0 > $R/ctp_basis_none_s0.txt
  traj --model ctp --output steps --condition atoms --out $R/ctp_steps_atoms_s0.pt --seed 0 > $R/ctp_steps_atoms_s0.txt
  ;;
baselines)
  traj --model mdn     --condition atoms --out $R/mdn_atoms_s0.pt     --seed 0 > $R/mdn_atoms_s0.txt
  traj --model anchors --condition atoms --out $R/anchors_atoms_s0.pt --seed 0 > $R/anchors_atoms_s0.txt
  traj --model ctp --output basis --condition text --out $R/ctp_basis_text_s0.pt --seed 0 > $R/ctp_basis_text_s0.txt
  traj --model ctp --output basis --condition text --init-sbm $R/sbm_jacktol_anyorder_s0.pt \
       --out $R/ctp_basis_textinit_s0.pt --seed 0 > $R/ctp_basis_textinit_s0.txt
  ;;
ablations)
  sbm --factorization independent --out $R/sbm_jacktol_independent_s0.pt --seed 0 > $R/sbm_jacktol_independent_s0.txt
  sbm --factorization mixture     --out $R/sbm_jacktol_mixture_s0.pt     --seed 0 > $R/sbm_jacktol_mixture_s0.txt
  sbm --factorization anyorder --no-history --out $R/sbm_jacktol_nohist_s0.pt --seed 0 > $R/sbm_jacktol_nohist_s0.txt
  traj --model ctp --output basis --condition atoms --time-sampling uniform \
       --out $R/ctp_basis_atoms_unif_s0.pt --seed 0 > $R/ctp_basis_atoms_unif_s0.txt
  ;;
seeds)
  for s in 1 2; do
    sbm  --factorization anyorder --out $R/sbm_jacktol_anyorder_s$s.pt --seed $s > $R/sbm_jacktol_anyorder_s$s.txt
    traj --model ctp --output basis --condition atoms --out $R/ctp_basis_atoms_s$s.pt --seed $s > $R/ctp_basis_atoms_s$s.txt
    traj --model ctp --output basis --condition none  --out $R/ctp_basis_none_s$s.pt  --seed $s > $R/ctp_basis_none_s$s.txt
  done
  ;;
channels)
  for p in us_atc_turbo_bench whisper_large_v3_bench; do
    py -m atc_forecast.data.messages --data data/v2 --profile $p
    sbm --profile $p --factorization anyorder --out $R/sbm_${p}_anyorder_s0.pt --seed 0 > $R/sbm_${p}_anyorder_s0.txt
  done
  ;;
eval)
  opt() { [ -f "$2" ] && echo "$1 $2" || true; }
  py -m atc_forecast.evaluation.compose_v2 --sbm $R/sbm_jacktol_anyorder_s0.pt \
     --ctp $R/ctp_basis_atoms_s0.pt --hist-only $R/ctp_basis_none_s0.pt \
     $(opt --direct $R/ctp_basis_text_s0.pt) $(opt --direct-init $R/ctp_basis_textinit_s0.pt) \
     $(opt --mdn $R/mdn_atoms_s0.pt) $(opt --anchors $R/anchors_atoms_s0.pt) \
     --split test --out results/v2/eval_test_s0.json \
     --per-message results/v2/eval_test_s0.npz ${W[@]+"${W[@]}"}
  py scripts/analyze_composition_v2.py --pm results/v2/eval_test_s0.npz --out results/v2/composition.json
  py scripts/collect_results_v2.py
  ;;
diagnose)
  py scripts/diagnose_v2.py nfe       --ctp $R/ctp_basis_atoms_s0.pt ${W[@]+"${W[@]}"}
  py scripts/diagnose_v2.py overfit   --ctp $R/ctp_basis_atoms_s0.pt ${W[@]+"${W[@]}"}
  py scripts/diagnose_v2.py hindsight --ctp $R/ctp_basis_atoms_s0.pt ${W[@]+"${W[@]}"}
  py scripts/diagnose_v2.py noise     --ctp $R/ctp_basis_atoms_s0.pt ${W[@]+"${W[@]}"}
  py scripts/diagnose_v2.py sbm       --sbm $R/sbm_jacktol_anyorder_s0.pt ${W[@]+"${W[@]}"}
  if [ -f $R/ctp_basis_atoms_unif_s0.pt ]; then
    py scripts/diagnose_v2.py time --ctp $R/ctp_basis_atoms_s0.pt $R/ctp_basis_atoms_unif_s0.pt ${W[@]+"${W[@]}"}
  fi
  py scripts/collect_results_v2.py
  ;;
sweeps)
  # 2000 test windows each; every sweep changes one factor of the main composition
  # (K=12 interpretations x S=6 futures, Euler NFE 12, main SBM, coefficient CTP).
  cmp() { local tag=$1; shift
    py -m atc_forecast.evaluation.compose_v2 --n 2000 --split test \
       --out results/v2/compose_$tag.json --run-name compose_$tag "$@" ${W[@]+"${W[@]}"}; }
  S0=$R/sbm_jacktol_anyorder_s0.pt; C0=$R/ctp_basis_atoms_s0.pt
  for ks in "1 72" "4 18" "12 6" "36 2"; do set -- $ks
    cmp K$1_S$2 --sbm $S0 --ctp $C0 --k $1 --s $2 --only oracle composed mode
  done
  cmp heun32 --sbm $S0 --ctp $C0 --method heun --nfe 32 --only oracle composed mode
  cmp ctp_steps --sbm $S0 --ctp $R/ctp_steps_atoms_s0.pt --only oracle composed mode
  for f in independent mixture nohist; do
    if [ -f $R/sbm_jacktol_${f}_s0.pt ]; then
      cmp sbm_$f --sbm $R/sbm_jacktol_${f}_s0.pt --ctp $C0 --only composed mode
    fi
  done
  py scripts/collect_results_v2.py
  ;;
*)
  sed -n '2,14p' "$0"; exit 1 ;;
esac
