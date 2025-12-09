#!/bin/bash
# Run solve agent on debug/test problems
# 3 problems, 5 minute time limit

cd "$(dirname "$0")/.."

python3 solve_agent.py \
    problems/debug_problems \
    --submissions_dir=submissions/debug \
    --time_limit_hours=0.083 \
    --model=qwen3-30b-a3b \
    --judge_model=qwen3-30b-a3b \
    --max_concurrent=8 \
    --target_perfect_scores=4
