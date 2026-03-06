#!/bin/bash
# RunPod Setup Script for Duelist Zero
# Run this after SSH-ing into your RunPod instance.
#
# Prerequisites:
#   - RunPod pod with A100 GPU + persistent volume mounted at /workspace
#   - Set your GitHub repo URL below (or use HTTPS with token)
#
# Usage:
#   curl -sSL <raw-github-url> | bash
#   # or just: bash scripts/runpod_setup.sh
set -e

# ============================================================
# Configuration — edit these if needed
# ============================================================
REPO_URL="https://github.com/STripV0/DuelistZero.git"
WORK_DIR="/workspace/duelist-zero"
N_ENVS=16          # More envs for A100 (adjust based on CPU cores)
TIMESTEPS=25000000

# ============================================================
# 1. System dependencies
# ============================================================
echo "=== Installing system dependencies ==="
apt-get update -qq
apt-get install -y -qq build-essential git wget > /dev/null 2>&1
echo "Done."

# ============================================================
# 2. Install uv (Python package manager)
# ============================================================
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # Make it persistent
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "uv version: $(uv --version)"

# ============================================================
# 3. Clone repo with submodules
# ============================================================
if [ ! -d "$WORK_DIR" ]; then
    echo "=== Cloning repository ==="
    git clone --recurse-submodules "$REPO_URL" "$WORK_DIR"
else
    echo "=== Repository exists, pulling latest ==="
    cd "$WORK_DIR"
    git pull
    git submodule update --init --recursive
fi
cd "$WORK_DIR"

# ============================================================
# 4. Download Lua 5.3.5 source (if not vendored)
# ============================================================
if [ ! -d "vendor/lua-5.3.5" ]; then
    echo "=== Downloading Lua 5.3.5 ==="
    cd vendor
    wget -q https://www.lua.org/ftp/lua-5.3.5.tar.gz
    tar xf lua-5.3.5.tar.gz
    rm lua-5.3.5.tar.gz
    cd ..
fi

# ============================================================
# 5. Build ygopro-core engine
# ============================================================
echo "=== Building ygopro-core ==="
bash build_core.sh

# ============================================================
# 6. Install Python dependencies
# ============================================================
echo "=== Installing Python dependencies ==="
uv sync

# ============================================================
# 7. Download card data (cards.cdb + Lua scripts)
# ============================================================
if [ ! -f "data/cards.cdb" ] || [ ! -d "data/script" ]; then
    echo "=== Downloading card data ==="
    uv run python scripts/download_data.py
fi

# ============================================================
# 8. Run tests to verify everything works
# ============================================================
echo "=== Running tests ==="
uv run pytest tests/ -x -q
echo ""

# ============================================================
# 9. Print GPU info
# ============================================================
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ============================================================
# Done!
# ============================================================
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo ""
echo " To resume training:"
echo "   cd $WORK_DIR"
echo "   nohup uv run python -m duelist_zero.training.self_play \\"
echo "     --timesteps $TIMESTEPS \\"
echo "     --n-envs $N_ENVS \\"
echo "     --pretrained-embeddings data/card_embeddings.npy \\"
echo "     --resume checkpoints/LATEST_CHECKPOINT.zip \\"
echo "     > training.log 2>&1 &"
echo ""
echo " To monitor training:"
echo "   tail -f training.log"
echo "   grep 'Eval.*Heuristic' training.log | tail -10"
echo ""
echo " To sync checkpoints back to your machine:"
echo "   # From your LOCAL machine, run:"
echo "   # scp -r root@<POD_IP>:$WORK_DIR/checkpoints/ ./checkpoints/"
echo ""
echo " To push code changes to GitHub:"
echo "   git add -A && git commit -m 'your message' && git push"
echo "============================================================"
