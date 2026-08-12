#!/bin/sh
# Download the MiniMax-H3 checkpoint tree from Hugging Face.
#
# Usage:
#   scripts/download_model.sh [MODEL_DIR] [--ref2va]
#
# MODEL_DIR defaults to ./MiniMax-H3 (the layout the CLI and the real-*
# tests expect). Downloads the FL2VA tree (~37 GiB) by default; pass
# --ref2va to also fetch the optional Ref2VA transformer (~62 GiB).
#
# Requires either the `huggingface_hub` Python package (preferred) or
# git + git-lfs. Both resume interrupted transfers.

set -e

MODEL_DIR="${1:-MiniMax-H3}"
REF2VA=0
if [ "$2" = "--ref2va" ]; then
    REF2VA=1
fi

REPO="MiniMaxAI/MiniMax-H3"

if command -v python3 >/dev/null 2>&1 && \
   python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "h3: downloading $REPO -> $MODEL_DIR (huggingface_hub)"
    python3 - "$MODEL_DIR" "$REF2VA" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

model_dir, ref2va = sys.argv[1], sys.argv[2] == "1"
allow = ["FL2VA/*"]
if ref2va:
    allow.append("Ref2VA/*")
print(f"h3: fetching {allow} (resumable)")
snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    local_dir=model_dir,
    allow_patterns=allow,
    max_workers=4,
)
print(f"h3: checkpoint ready under {model_dir}")
PYEOF
else
    echo "h3: huggingface_hub not found; falling back to git-lfs"
    if ! command -v git >/dev/null 2>&1 || \
       ! git lfs version >/dev/null 2>&1; then
        echo "h3: need 'huggingface_hub' (pip install huggingface_hub) or git-lfs" >&2
        exit 1
    fi
    if [ ! -d "$MODEL_DIR/.git" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/$REPO" "$MODEL_DIR"
    fi
    if [ "$REF2VA" = "1" ]; then
        (cd "$MODEL_DIR" && git lfs pull --include="FL2VA/**,Ref2VA/**")
    else
        (cd "$MODEL_DIR" && git lfs pull --include="FL2VA/**")
    fi
    echo "h3: checkpoint ready under $MODEL_DIR"
fi

if [ -f "$MODEL_DIR/FL2VA/tokenizer/tokenizer.json" ]; then
    echo "h3: FL2VA tree looks complete (tokenizer present)"
else
    echo "h3: warning: FL2VA/tokenizer/tokenizer.json missing" >&2
fi
