"""Merge a teacher adapter into the base model and save the result (weights in bf16, tokenizer included),
so that the trained teacher can be served by vLLM for synthesis.

  python rodel/merge_lora.py --adapter runs/seq/ckpt/t1/epoch1/t --out runs/seq/ckpt/t1/merged
"""
import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_MODEL

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print("merged ->", a.out)
