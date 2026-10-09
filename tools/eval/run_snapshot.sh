#!/usr/bin/env bash
# Non-thinking capability snapshot (lm-eval 0.4.x). Not a leaderboard run: it is a
# regression baseline you re-run after every engine/patch change, on the same box,
# with the same extractor, and compare against results/nonthinking_snapshot.txt.
#
# Thinking is switched off by tools/eval/nt_proxy.py, because lm-eval cannot send
# chat_template_kwargs. Start it first:  python nt_proxy.py &   (listens on :8001)
#
# usage: run_snapshot.sh [out_dir]      ~2.5 h total at 4-way concurrency
set -u
S=${1:-./snapshot_$(date +%Y%m%d-%H%M)}; mkdir -p "$S"
MODEL=${MODEL:-qwen3.8-flash-next}
TOK=${TOKENIZER:?set TOKENIZER=/path/to/model/dir}
CHAT="model=$MODEL,base_url=http://127.0.0.1:8001/v1/chat/completions,num_concurrent=4,max_retries=3,timeout=900"
COMP="model=$MODEL,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=$TOK,tokenized_requests=True,num_concurrent=4,max_retries=3"
run(){ local name=$1; shift; echo "$(date +%T) START $name"; "$@" > "$S/$name.log" 2>&1; echo "$(date +%T) END $name rc=$?"; }
run ifeval     lm_eval --model local-chat-completions --model_args "$CHAT" --tasks ifeval --apply_chat_template --gen_kwargs max_gen_toks=2048,temperature=0 --output_path "$S/out_ifeval" --log_samples
run mgsm_zh_ja lm_eval --model local-chat-completions --model_args "$CHAT" --tasks mgsm_native_cot_zh,mgsm_native_cot_ja --apply_chat_template --gen_kwargs max_gen_toks=1536,temperature=0 --output_path "$S/out_mgsm" --log_samples
run truthfulqa lm_eval --model local-completions --model_args "$COMP" --tasks truthfulqa_mc2,truthfulqa_zh_mc2 --output_path "$S/out_tqa"
run mmlu_pro   lm_eval --model local-chat-completions --model_args "$CHAT" --tasks mmlu_pro --limit 25 --apply_chat_template --gen_kwargs max_gen_toks=2048,temperature=0 --output_path "$S/out_mmlu_pro" --log_samples
