# Duelist Zero: GOAT Edition

> **Work in progress** — actively under development, expect breaking changes.

A reinforcement learning agent that learns to play Yu-Gi-Oh! in the GOAT format (April 2005 cardpool/banlist). Built from scratch with Python ctypes bindings to the [ygopro-core](https://github.com/Fluorohydride/ygopro-core) C++ engine.

The agent uses MaskablePPO (masked action-space PPO) with self-play training to learn game strategy from zero domain knowledge.

GOAT format is the starting point. Tthe long-term goal is to extend Duelist Zero to the current TCG/OCG format with the full modern cardpool.

## Setup

**Prerequisites:** Linux (or WSL2), a C++ compiler (g++), Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/STripV0/DuelistZero.git
cd duelist-zero

# Download Lua 5.3.5 (required by ygopro-core)
cd vendor
wget https://www.lua.org/ftp/lua-5.3.5.tar.gz
tar xf lua-5.3.5.tar.gz
cd ..

# Build the engine
./build_core.sh    # produces lib/libocgcore.so

# Install Python deps
uv sync

# Download card database + scripts (required)
uv run python scripts/download_data.py
```

## Play Against It (EDOPro)

You can duel the trained agent in [EDOPro](https://projectignis.github.io/):

1. Open EDOPro and host a LAN duel (GOAT format, default port `7911`)
2. In a terminal, run:
   ```bash
   uv run python scripts/edopro_bot.py
   ```
3. The bot connects as "DuelistZero" and plays using the trained model

Options:
```bash
# Use a specific model checkpoint
uv run python scripts/edopro_bot.py --model checkpoints/ckpt_00500000

# Connect to a different host/port
uv run python scripts/edopro_bot.py --host 192.168.1.10 --port 7911

# Use a different deck
uv run python scripts/edopro_bot.py --deck data/deck/Chaos.ydk
```

> A pre-trained model will be provided once training stabilizes. For now, you'll need to train your own (see below).

## Training (Optional)

```bash
# Run tests
uv run pytest tests/

# Smoke test (single duel)
uv run python scripts/smoke_test.py

# Train (self-play with MaskablePPO)
uv run python -m duelist_zero.training.self_play --timesteps 2000000
```

## Project Structure

```
src/duelist_zero/
  core/       ctypes bindings to libocgcore.so, callbacks, message parser
  engine/     Duel lifecycle, GameState tracking
  env/        GoatEnv (Gymnasium), action masking, observation encoder, rewards
  network/    Custom feature extractor (card embeddings)
  training/   Self-play pipeline, ELO tracking, curriculum learning
data/         Card database (.cdb), Lua scripts, banlist, deck files (.ydk)
vendor/       ygopro-core (git submodule) + Lua 5.3.5 (downloaded)
scripts/      Debugging and utility scripts
tests/        pytest suite
```

## How It Works

1. **Engine** — ygopro-core handles all game rules, card effects, and state transitions via a C++ shared library
2. **Environment** — `GoatEnv` wraps the engine as a Gymnasium env with a 71-action discrete space, dict observations (329-dim features + 30-dim card IDs), and action masking for legal moves
3. **Training** — MaskablePPO with self-play: the agent trains against a mix of its current self (70%), past checkpoints (15%), and a heuristic opponent (15%), with curriculum deck scheduling
4. **Rewards** — Terminal win/loss (+1/-1), LP-delta shaping, card advantage shaping, step penalty to discourage stalling

## Tech Stack

- **Engine:** ygopro-core (C++) with Lua 5.3 scripting
- **Bindings:** Python ctypes
- **RL:** Stable-Baselines3 + sb3-contrib (MaskablePPO)
- **Environment:** Gymnasium
- **Training:** Self-play with ELO tracking and curriculum learning
