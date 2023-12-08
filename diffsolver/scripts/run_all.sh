#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python scripts/run_all_prog.py
python scripts/run_all_lang.py
python scripts/run_all_difflang.py
python scripts/run_all_toollang.py
python scripts/run_all_sac.py
