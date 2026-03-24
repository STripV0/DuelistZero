# Duelist Zero — Technical Summary

## Algorithm

**MaskableRecurrentPPO** — Custom algorithm merging sb3-contrib's RecurrentPPO (LSTM memory) + MaskablePPO (action masking). LSTM tracks temporal state across game decisions within an episode.

| Parameter | Value |
|---|---|
| Learning rate | 3e-5 |
| n_steps (rollout) | 512 |
| Batch size | 256 |
| n_epochs | 4 |
| Gamma | 0.99 |
| Entropy coef | 0.05 |
| VF coef | 0.8 |
| Clip range | 0.2 |
| Parallel envs | 16 (SubprocVecEnv) |
| LSTM hidden size | 256 |
| LSTM layers | 1 |

## Neural Network Architecture

```
Observation Dict
  ├── "features"         (462,)    float32 — game state features
  ├── "card_ids"         (50,)     float32 — card identity indices
  ├── "action_features"  (71, 29)  float32 — rich per-action features with ownership,
  │                                         effect categories, and ATK matchup
  └── "action_history"   (16, 13)  float32 — structured action history with outcomes

CardEmbeddingExtractor (Three-Segment Transformer):

  nn.Embedding(vocab_size=~14k, embed_dim=64, padding_idx=0)
    └── optionally initialized from pretrained embeddings (.npy)

  Board tokens:   card_ids(50) → embed(64) → Identity (64=d_model) → (50, 64)
  Action tokens:  action_features(71,29) → embed col0(64) + cols1-28 → Linear(92, 64)  → (71, 64)
  History tokens: action_history(16,13)  → embed col0(64) + cols1-12 → Linear(76, 64)  → (16, 64)

  + segment_embedding(3, 64): board=0, action=1, history=2
  → concat: (137, 64)
  → TransformerEncoder(d_model=64, heads=4, layers=2)
  → segment-aware masked mean pooling
  → board_pool(64) | action_pool(64) | history_pool(64)
  → Linear(192, 256) + ReLU → embed_stream(256)

  Side outputs (padded positions zeroed):
    _last_action_tokens: (B, 71, 64) for cross-attention action head
    _last_board_tokens:  (B, 50, 64) for cross-attention value head

  Merge:
    [features(462) | embed_stream(256)] = 718
    → Linear(718, 256) → ReLU
    → Linear(256, 256) → ReLU → LSTM(256) →
                           │
                  ┌────────┴────────┐
             Policy Head       Value Head
          Linear(512, 256)   Linear(512, 256)
                  │                  │
     CrossAttentionActionHead   Linear(256, 1):
       action_tokens(71, 64)      latent_vf(256) from LSTM+MLP
       latent_pi(256)             → scalar V(s)
       → cat per action (320)
       → Linear(320,64)→ReLU
       → Linear(64,1)→(71,)
```

### action_features — (71, 29) per action slot

| Col | Feature | Range | Phase |
|---|---|---|---|
| 0 | card_id (embedding index) | [0, vocab) | original |
| 1 | is_summon | {0,1} | original |
| 2 | is_spsummon | {0,1} | original |
| 3 | is_set | {0,1} | original |
| 4 | is_activate | {0,1} | original |
| 5 | is_attack | {0,1} | original |
| 6 | is_reposition | {0,1} | original |
| 7 | is_phase/pass/other | {0,1} | original |
| 8 | ATK / 5000 | [0,1] | original |
| 9 | DEF / 5000 | [0,1] | original |
| 10 | level / 12 | [0,1] | original |
| 11 | location | 0.0=none, 0.2=hand, 0.4=mzone, 0.6=szone, 0.8=grave, 1.0=removed | original |
| 12 | target_is_mine | {0,1} — SELECT_CARD/TRIBUTE/CHAIN only | C1 |
| 13 | target_is_opp | {0,1} — SELECT_CARD/TRIBUTE/CHAIN only | C1 |
| 14 | destroys_monster | {0,1} | C2 |
| 15 | destroys_spelltrap | {0,1} | C2 |
| 16 | negates | {0,1} | C2 |
| 17 | draws_cards | {0,1} | C2 |
| 18 | searches_deck | {0,1} | C2 |
| 19 | flips_facedown | {0,1} | C2 |
| 20 | burns_lp | {0,1} | C2 |
| 21 | bounces_to_hand | {0,1} | C2 |
| 22 | special_summons | {0,1} | C2 |
| 23 | gains_atk | {0,1} | C2 |
| 24 | changes_position | {0,1} | C2 |
| 25 | protects | {0,1} | C2 |
| 26 | target_atk_norm (strongest opp monster ATK/5000) | [0,1] — attack actions only | C3 |
| 27 | atk_advantage (my ATK - max opp ATK)/5000 | [-1,1] — attack actions only | C3 |
| 28 | opp_has_facedown | {0,1} — attack actions only | C3 |

### action_history — (16, 13) last 16 public actions

| Col | Feature | Range |
|---|---|---|
| 0 | card_id (embedding index) | [0, vocab) |
| 1 | player (0=me, 1=opp) | {0,1} |
| 2 | is_summon | {0,1} |
| 3 | is_set | {0,1} |
| 4 | is_activate | {0,1} |
| 5 | is_attack | {0,1} |
| 6 | is_draw | {0,1} |
| 7 | ATK / 5000 | [0,1] |
| 8 | DEF / 5000 | [0,1] |
| 9 | turns_ago / 40 | [0,1] |
| 10 | damage / 8000 (battle damage dealt) | [0,1] |
| 11 | cards_destroyed / 5 (by this action) | [0,1] |
| 12 | was_negated | {0,1} |

### LLM-Pretrained Card Embeddings

Card embeddings can be pre-initialized from card text using sentence-transformers:

1. Card name + effect text encoded with `all-MiniLM-L6-v2` (384-dim)
2. PCA reduction to 64 dims (matching `embed_dim`, 64.4% variance explained)
3. Saved as `data/card_embeddings.npy` (14,337 × 64, index 0 = zero padding)
4. Loaded at model init, then fine-tuned during training

Generate: `uv run --group embeddings python scripts/generate_card_embeddings.py`
Train with: `--pretrained-embeddings data/card_embeddings.npy`

## Observation Space (462 features + structured keys)

### features (462,)

| Range | Description |
|---|---|
| [0-9] | Scalars: LP, hand size, deck count, GY count, banished (×2 players, normalized) |
| [10-17] | Phase one-hot (8 phases) |
| [18] | Turn player flag |
| [19] | Turn count / 40 |
| [20-22] | Relative advantage: LP diff, monster count diff, ATK diff ([-1,1]) |
| [23-102] | My monster zone: 5 slots × 16 features |
| [103-152] | My spell/trap zone: 5 slots × 10 features |
| [153-232] | Opp monster zone: 5 slots × 16 features (face-down stats hidden) |
| [233-282] | Opp spell/trap zone: 5 slots × 10 features (face-down hidden) |
| [283-432] | My hand: 10 slots × 15 features |
| [433-452] | Deck identity one-hot (20) |
| [453-461] | Opponent inference: extra draws, facedown count, 5×S/T age, GY monsters, banished |

### card_ids (50,)

My field (10) + opp field face-up (10) + my hand (10) + my GY top 10 + opp GY top 10.

### action_features (71, 29) and action_history (16, 13)

See architecture section above for full column layouts.

## Action Space (71 discrete actions)

| Range | Action |
|---|---|
| 0-4 | Summon monster 0-4 |
| 5-9 | Special summon 0-4 |
| 10-14 | Set monster 0-4 |
| 15-19 | Set spell/trap 0-4 |
| 20-29 | Activate card 0-9 |
| 30-34 | Reposition monster 0-4 |
| 35 | Go to Battle Phase (masked when no ATK-position monsters) |
| 36 | Go to End Phase |
| 37-41 | Attack with monster 0-4 |
| 42-46 | Battle activate 0-4 |
| 47 | Go to Main Phase 2 |
| 48 | Go to End Phase (battle) |
| 49-50 | Yes / No |
| 51-60 | Select card 0-9 (sequential multi-select for 2+ cards) |
| 61-62 | Position ATK / DEF |
| 63-70 | Select option 0-7 |

### Sequential Multi-Card Selection

When the engine asks for 2+ cards (tributes, Graceful Charity discards, etc.), the agent picks one card per step instead of auto-filling from a single index. A state machine in `GoatEnv` tracks:

- `_multi_select_indices`: cards picked so far
- `_multi_select_needed`: total cards required (from `msg.min_count`)
- `_multi_select_msg`: original SELECT_CARD/SELECT_TRIBUTE message

The action mask excludes already-selected cards. Cancel (action 50) is available mid-selection if the message is cancelable. Intermediate picks return reward=0; the full reward is computed when all picks are collected and sent to the engine. Opponents still use the old auto-fill path.

## Reward Function

Terminal reward scaled by turn count, plus potential-based reward shaping (PBRS).

### Terminal Reward

| Outcome | Reward |
|---|---|
| Win (turn ≤ 5) | 1.0 |
| Win (turn 5-20) | Linear interpolation 1.0 → 0.3 |
| Win (turn ≥ 20) | 0.3 |
| Loss | -1.0 |
| Draw | 0.0 |
| Truncation (200 steps) | -1.0 |

### Potential-Based Reward Shaping (PBRS)

PBRS (Ng et al. 1999) adds intermediate reward signal without changing the optimal policy. Each step receives a shaping bonus:

```
F(s, s') = shaping_scale × (γ · Φ(s') − Φ(s))
```

The potential function `Φ(s)` uses a hybrid of ATK-weighted board power, card count, and LP:

```python
Φ(s) = 0.40 × atk_advantage + 0.25 × card_advantage + 0.35 × lp_advantage
```

- `atk_advantage`: `clamp(sum(my_faceup_atk/5000) - sum(opp_faceup_atk/5000), -1, 1)` — ATK-weighted board power (requires card DB). Falls back to count-based `(my_monsters - opp_monsters) / 5` when DB unavailable.
- `card_advantage`: `clamp((my_total_cards - opp_total_cards) / 10, -1, 1)` — total cards (hand + field)
- `lp_advantage`: `clamp((my_lp - opp_lp) / 8000, -1, 1)`

The ATK-weighted approach means tributing 2 weak monsters (1000 ATK each) for 1 strong monster (2800 ATK) registers as a net improvement, unlike the old count-based board_adv which would see it as -1 card. Card advantage is kept so Pot of Greed (+2 cards) and setting traps still register as positive potential.

Terminal states use `Φ = 0` (absorbing state convention). With `shaping_scale=0.5`, shaped rewards are ~[-0.02, 0.02] per step vs terminal [0.3, 1.0] / -1.0.

Configure via `--shaping-scale` (default 0.5, set to 0 to disable).

## Training Strategy

**Phase 1 — Pre-training:** Agent trains vs heuristic opponent only (diverse decks from pool). Evaluates every 50k steps.

**Phase 2 — Frozen-pool self-play** (activates when eval ≥ 85% vs heuristic):
- 60% heuristic opponent (diverse deck from pool)
- 20% frozen recent checkpoint (mirror deck)
- 20% frozen older checkpoint (mirror deck)
- **Regression gate at 60%:** deactivates self-play if heuristic WR drops below
- Per-episode opponent roll (consistent opponent within each game)

All 7 decks available from start (no curriculum). Heuristic games use diverse decks, checkpoint games use mirror matches.

## Baseline Opponent (Heuristic)

Mask-based priority without using observations:
- Idle: Summon > SpSummon > Battle Phase > Set Monster > End Phase
- Battle: Attack > Main2 > End
- Chain: Pass unless forced
- Position: ATK > DEF
- Select: First card, prefer selecting over cancelling

---

# Issue History

## Session 1 — Feb 20 (Initial Training Pipeline Build)

### Issue 1: Tensorboard not installed
**Problem:** Training script crashed because tensorboard was not installed.
**Fix:** Made tb_log conditional — training skips tensorboard logging if the package is not available.

### Issue 2: "No pending decision" assertion after opponent swap
**Problem:** After swapping the training opponent, `GoatEnv.reset()` hit an assertion error because the game ended before the agent got a turn.
**Fix:** Made `reset()` retry up to 10 times if the game ends before the agent gets a decision. Made `step()` handle `None` pending_msg gracefully.

### Issue 3: 100% win rate plateau — agent "too good" immediately
**Problem:** After 500k steps, agent achieved 100% win rate vs random in only ~5-8 steps per game. ELO plateaued around 1200-1228.
**Root cause:** Not discovered until Session 4 — the TAG_MODE bug (issue #10).

## Session 2 — Feb 20 (Initial Exploration)
No bugs. Short session for project exploration.

## Session 3 — Feb 21 (EDOPro Bot + Network Protocol)

### Issue 4: EDOPro CTOS_JoinGame version2 field missing
**Problem:** Bot connected to EDOPro but nothing showed up in the game lobby. Missing `version2` field (`ClientVersion` struct).
**Fix:** Added `ClientVersion` struct with `client_major=41, client_minor=0, core_major=11, core_minor=0`.

### Issue 5: EDOPro struct alignment padding
**Problem:** After adding `version2`, bot still failed — `network.h` has no `#pragma pack`, adding 2 bytes padding between `version` (u16) and `gameid` (u32).
**Fix:** Changed `struct.pack("<HI")` to `struct.pack("<HHI")` with explicit padding.

### Issue 6: Deck sent after READY instead of before
**Problem:** Bot sent `CTOS_HS_READY` without sending the deck first. EDOPro requires deck before ready.
**Fix:** Reordered to send deck before `CTOS_HS_READY`.

### Issue 7: DeckError struct parsing at wrong offsets
**Problem:** `DeckError` response (24 bytes with alignment) was read at wrong offsets.
**Fix:** Fixed struct parsing to properly read `MAINCOUNT` error.

### Issue 8: Card database mismatch (EDOPro vs local cards.cdb)
**Problem:** EDOPro uses different card codes for some cards (e.g., `504700178` for Sangan instead of `26202165`). One card classified as extra deck instead of main deck.
**Fix:** Mapped 11 EDOPro-specific card IDs to standard Konami IDs.

## Session 4 — Feb 23 (TAG_MODE Discovery — CRITICAL)

### Issue 9: EDOPro message format differences (u8 vs u32 counts)
**Problem:** EDOPro's ygopro-core uses completely different message formats: `u32` counts (not `u8`), `u64` descriptions (not `u32`), `u32` sequences, `loc_info` structs. Bot crashed on `_parse_select_battle_cmd`.
**Fix:** Updated the network bot's parser to handle EDOPro's extended format for all SELECT_* messages.

### Issue 10: DUEL_1_FACEUP_FIELD = 0x20 was actually DUEL_TAG_MODE (CRITICAL)
**Problem:** Python constants for duel options were completely wrong:
- `DUEL_1ST_TURN_DRAW = 0x10` was actually `DUEL_PSEUDO_SHUFFLE`
- `DUEL_1_FACEUP_FIELD = 0x20` was actually `DUEL_TAG_MODE`
- `GOAT_DUEL_OPTIONS = 0x38` accidentally enabled Tag Duel mode
- On turn 2, the engine swapped the opponent's deck with an empty "tag deck," causing instant deck-out
- The agent learned "just end turn" = free win, achieving 100% win rate in ~5 steps

**Fix:** Removed incorrect constants. GOAT format only needs `DUEL_OBSOLETE_RULING = 0x08`.
**Impact:** Root cause of issue #3. All previous training was invalid.

### Issue 11: Random-vs-random always ends by deck-out
**Problem:** After fixing TAG_MODE, random-vs-random always ended by deck-out (~200 steps, no combat). Player 0 always lost (drew first). Sparse reward gave zero learning signal.
**Fix:** Introduced heuristic exploration policy, LP-delta reward shaping, and player order randomization.

### Issue 12: Eval asymmetry — opponents not deterministic
**Problem:** Self-play opponents used `deterministic=False` during eval, adding noise to win rate measurements.
**Fix:** `deterministic=True` during eval, `deterministic=False` for training opponents.

## Session 5 — Feb 23 (Card Embeddings + EDOPro Bot Fixes)

### Issue 13: EDOPro bot player index mapping wrong
**Problem:** Bot was assigned as player 1 from lobby slot, but engine uses different player numbers. When going first, `SELECT_IDLECMD` came for player 0, bot skipped it → timeout.
**Fix:** Fixed `my_player` mapping from lobby slot to engine player index.

### Issue 14: Bot enters battle phase but never attacks
**Problem:** Trained model learned to summon and enter battle but used action 42 (`_BATTLE_MAIN2`) instead of attack actions.
**Root cause:** Training maturity issue — all previous training was on TAG_MODE-bugged engine.

### Issue 15: EDOPro SELECT_CARD response format
**Problem:** Bot crashed on `SELECT_CARD` — response format unclear (1-byte vs 4-byte indices).
**Fix:** Confirmed `CTOS_RESPONSE` goes through `set_responseb`: `count(1B) + indices(1B each)`.

### Issue 16: Dragons goat.ydk has EDOPro pre-errata card codes
**Problem:** "Dragons goat.ydk" had 4 cards not in local DB (codes with `5xxxxxxxx` prefix).
**Fix:** Swapped pre-errata codes for standard IDs.

## Session 6 — Feb 23 (Heuristic + Infinite Loop Debugging)

### Issue 17: Heuristic Activate priority causes infinite loop
**Problem:** Heuristic kept activating the same card effect repeatedly, causing infinite loop. Training hung at 100% CPU.
**Fix:** Removed Activate from idle/battle heuristic priority. Heuristic now only summons, attacks, sets, ends.

### Issue 18: Safety counter insufficient for infinite loops
**Problem:** Even with heuristic fix, 5/900 games hung. 200-decision safety counter not enough.
**Fix:** Added `_MAX_ADVANCE_ITERS`, then 2-second `time.monotonic()` timeout in `Duel.process()`.

### Issue 19: Python stdout buffering when piped to file
**Problem:** Training output invisible — zero output in log file.
**Fix:** Used `PYTHONUNBUFFERED=1`.

### Issue 20: `Duel.process()` while-True loop hangs
**Problem:** Root cause of hangs: `process()` had a `while True` loop. Engine enters states where it processes without producing response messages or ending.
**Fix:** 2-second timeout inside `Duel.process()` that forces `is_finished=True`.

### Issue 21: Self-play passivity (mirror-match truces)
**Problem:** When self-play activated, episode lengths doubled (50→110 steps). Both sides played same cautious policy, creating passive standoffs.
**Fix:** Changed opponent mix to 40/40/20 (self/heuristic/past). Added aggression bonus for battle damage.

## Session 7 — Feb 23 (Self-Play Deck Issue)

### Issue 22: Self-play opponent gets wrong deck
**Problem:** Checkpoint opponent received random deck from pool but was trained on Goat Control only. Saw unfamiliar card IDs, acted semi-randomly.
**Recommendation:** Force opponent to also use Goat Control during self-play.

## Session 8 — Feb 24 (Training Pipeline Overhaul)

### Issue 23: HeuristicExplorationWrapper breaks PPO
**Problem:** Epsilon-greedy wrapper injected off-policy random actions. PPO is on-policy — this mathematically breaks gradient updates.
**Fix:** Deleted `HeuristicExplorationWrapper`. Increased `ent_coef` from 0.01 to 0.05 for native exploration.

### Issue 24: batch_size=64 too small
**Problem:** With 329-dim obs, batch_size=64 caused noisy gradients and wild 30-70% win rate oscillations.
**Fix:** Increased to 512 (2048/512 = 4 minibatches).

### Issue 25: Step penalty causing "die fast" local optimum
**Problem:** Step penalty of -0.002 × ~130 steps = -0.26 per episode dominated reward. Agent learned to "die fast" to minimize penalty. Episode lengths dropped without win rate improving.
**Fix:** Set step penalty to 0.0.

### Issue 26: Self-play opponent swaps too frequent
**Problem:** Opponent swapped every 20k steps to random checkpoint from entire history. Agent chased moving target, forgot how to beat previous opponents (26-44% vs old checkpoints).
**Fix:** Gated self-play behind 3 consecutive evals at 80%+. Mixed opponent: 70% current self, 15% heuristic, 15% recent past.

### Issue 27: FPS death spiral from synchronous evaluation
**Problem:** FPS dropped from 132 to 31. Eval ran synchronously every 20k steps, ~75% wall time spent evaluating.
**Fix:** Checkpoint interval 20k→50k, eval episodes 50→20, removed pool eval during pre-training.

### Issue 28: No truncation penalty
**Problem:** 11/20 games truncated at 500 steps with no penalty, allowing stalling to draws.
**Fix:** Reduced `_MAX_STEPS` 500→200, added truncation penalty of -1.0.

### Issue 29: `card_count()` weights hand = field equally
**Problem:** Card advantage reward counted hand and field equally, incentivizing hoarding cards.
**Fix:** Weighted field cards 1.5x higher than hand cards.

### Issue 30: Self-play gated at 80% but never reached
**Problem:** Self-play activation required 80% WR, which was never achieved. Agent stuck training vs heuristic forever.
**Fix:** Lowered threshold to 60%.

### Issue 31: Chain-pass spam eats 65% of agent steps
**Problem:** Agent bombarded with `MsgSelectChain` where only valid action was "pass" — 424/917 decisions were forced no-choice passes. Burned through 200-step limit.
**Fix:** Auto-respond to forced single-option decisions in `_advance()` without counting as agent steps.

### Issue 32: Set-tribute cancel loop
**Problem:** Agent tried to set high-level monster, got tribute prompt, cancelled, picked same set action → 100+ step infinite loop while having lethal on board.
**Fix:** Track cancelled tribute card code, mask out that card's summon/set on next idle cmd via `_cancelled_tribute_code`.

### Issue 33: No action-to-card mapping (fundamental representation gap)
**Problem:** Observation showed board state and mask showed valid indices, but nothing connected them. "Attack with monster 0 in attackable list" — agent had no idea which field monster that was.
**Fix:** Added `encode_action_cards()` + `ACTION_CARD_DIM = 71`. Maps each action slot to its card's embedding index. Added `"action_cards"` to observation space.

## Session 9 — Feb 24 (Curriculum Fixes)

### Issue 34: Plateau detection statistically impossible
**Problem:** Used `max(recent_5) - min(recent_5) < 0.02`. With 10-episode evals, win rates quantized to 10% increments — plateau never triggers, curriculum never advances.
**Fix:** Replaced with improvement-based: `late_avg - early_avg < threshold` over `2 × plateau_window` evals. Increased eval episodes to 50.

### Issue 35: `self.self_play_window = 2` hardcoded bug
**Problem:** Line 44 had `self.self_play_window = 2` hardcoded instead of using constructor parameter.
**Fix:** Changed to `self.self_play_window = self_play_window`.

### Issue 36: Heuristic can't pilot combo decks
**Problem:** Heuristic plays "summon → attack → end" which misplays combo decks like Empty Jar. Eval vs heuristic meaningless at late curriculum stages.
**Fix:** When self-play active with ≥2 checkpoints, eval vs sampled past checkpoint. Use that WR for curriculum progression.

### Issue 37: No deck identity signal
**Problem:** During self-play with diverse decks, agent/opponent had no way to know which deck they were playing.
**Fix:** Added 20-dim one-hot deck identity to observation. OBSERVATION_DIM: 329 → 349.

## Session 10 — Feb 25 (Further Training Fixes)

### Issue 38: ZoneCard import inside if-block
**Problem:** `ZoneCard` import inside conditional block caused `UnboundLocalError`.
**Fix:** Moved import to top of file.

### Issue 39: SIGPIPE from `| head -30` killing training
**Problem:** Running training with `| head -30` caused SIGPIPE, killing the process after ~100k steps.
**Fix:** Ran training via `nohup` without piping.

## Session 11 — Feb 25 (CardData Struct + Battle Categories — CRITICAL)

### Issue 40: CardData struct setcode field completely wrong (CRITICAL)
**Problem:** Python ctypes `CardData` struct had `setcode` as `c_uint32 * 4` (16 bytes), but C struct uses `uint16_t setcode[16]` (32 bytes). 16-byte mismatch shifted every field after — engine saw `type=0` for all cards.
**Fix:** Changed `("setcode", c_uint32 * 4)` to `("setcode", c_uint16 * 16)`.
**Impact:** Engine thought every card had `type=0`, never offered summon. Agent trained for hundreds of thousands of steps only seeing "pass chain", "end turn", "set spell/trap". All prior training fundamentally broken.

### Issue 41: PROCESSOR_WAITING / PROCESSOR_END constants wrong
**Problem:** Python said `PROCESSOR_WAITING = 0x1`, `PROCESSOR_END = 0x2`. Engine defines `0x10000000` and `0x20000000`. Process method never detected waiting/end states properly.
**Fix:** Fixed constants and flag checking.

### Issue 42: Battle command categories swapped (CRITICAL)
**Problem:** Attack was category 0 and activate was category 1 — reversed from engine (0=activate, 1=attack). Every attack was interpreted as "activate effect" and vice versa.
**Fix:** Swapped categories to match engine.
**Impact:** Caused both hangs (invalid index) and inability to attack.

### Issue 43: Heuristic tribute cancel loop
**Problem:** Heuristic fallback `valid[0]` picked "cancel" for `SELECT_TRIBUTE`, creating summon→cancel→summon infinite loop.
**Fix:** Prefer selecting cards over cancelling in tribute prompts.

### Issue 44: CONFIRM_DECKTOP/EXTRATOP parser format
**Problem:** Used `CONFIRM_CARDS` parser which reads extra `skip_panel` byte. Corrupted buffer position for all subsequent messages.
**Fix:** Added separate parsers without `skip_panel`.

### Issue 45: C engine hangs on Lua infinite loops
**Problem:** With CardData fix, card effects worked but some caused `core.process()` to hang — Lua scripts entered infinite loops.
**Attempted:** Lua instruction hook, C++ iteration limit, threading with GIL, SIGALRM.
**Fix:** Threading wrapper where each `process()` runs in daemon thread with 2-second timeout. Stuck games abandoned as draws. 500-decision loop guard in `_advance()`.

### Issue 46: Heuristic attack-cancel loop
**Problem:** "Prefer pass for chains" check fired on cancelable `SELECT_CARD` for attack targets. Attack → select target → cancel → attack → infinite loop.
**Fix:** Added `is_chain` parameter. Only `SELECT_CHAIN` triggers "prefer pass."

### Issue 47: MSG 140 (ANNOUNCE_RACE) unhandled
**Problem:** `ANNOUNCE_RACE` fell through to `respond_int(0)`, which was invalid.
**Fix:** Added `MsgAnnounceRace` parser. Auto-responds with first valid race bitmask.

### Issue 48: Special summons missing from action space
**Problem:** Action space mapped 7/8 idle categories but missed spsummon (category 1). Agent could never special summon BLS, Chaos Sorcerer, etc.
**Fix:** Added spsummon range to action space.

### Issue 49: Agent converges on passive "say no to everything"
**Problem:** Agent consistently converged on passive policy: 0 summons, 0 attacks, "no" 65.6%, END_PHASE 8.5%. Won 30% purely from heuristic limitations.
**Root cause:** Combination of CardData bug (#40) preventing summons, battle categories swapped (#42), and weak reward signal.
**Fix:** All underlying bugs fixed. Final training showed active combat.

### Issue 50: Agent attacks but doesn't deal damage
**Problem:** Agent attacked 195-229 times per game but LP barely changed — attacking into stronger monsters or defense positions it couldn't overcome.
**Diagnosis:** Mechanical "summon and attack" without regard to ATK/DEF values.

## Session 12 — Feb 25-26 (Reward + Parallel Envs + Self-Play)

### Issue 51: Reward confusion from 5 competing signals
**Problem:** Agent stuck at ~50% vs heuristic after 800k steps. 5 reward signals (LP delta, card advantage, aggression bonus, step penalty, terminal) pulled in different directions.
**Fix:** Replaced with sparse terminal reward (turn-count-scaled). Only: win fast = good, lose = bad.

### Issue 52: Low throughput (single env, 115 FPS)
**Fix:** SubprocVecEnv with 8 parallel environments → 300-400 FPS.

### Issue 53: PPO instability with parallel envs
**Problem:** Plateaued at 42-52%. clip_fraction: 0.54, approx_kl: 0.10, entropy collapsed.
**Root cause:** LR 3e-4 and 10 epochs too aggressive for 8× more data per rollout.
**Fix:** LR → 1e-4, epochs → 4, ent_coef → 0.10.

### Issue 54: Self-play collapse (72% → 36% vs heuristic)
**Problem:** Self-play activated at 650k, heuristic WR crashed to 36%. Agent became passive.
**Root cause:** Co-adaptation loop — both copies exploited each other's passivity.
**Fix:** Disabled self-play, resumed from 600k. Trained heuristic-only to 77%.

### Issue 55: Heuristic ceiling (~77%)
**Problem:** Heuristic-only plateaued at 74-80%. No pressure to learn deeper strategy.

### Issue 56: Round robin draws (22-33%)
**Problem:** Agent-vs-agent games frequently stalled into board locks.

### Issue 57: Self-play with regression gate still eroding (70% → 60%)
**Problem:** Second self-play attempt with 55% gate. WR drifted down over 3.5M steps. Vs-past WR stuck at 28-46%.
**Root cause:** 40% current-self opponent enabled co-adaptation. Gate too low at 55%.
**Fix:** Frozen opponent pool (no current-self). Gate raised to 70%.

### Issue 58: Observation blind spots (can't play from behind)
**Problem:** Agent plays well ahead, zero comeback ability. Goes to battle with no monsters. Stalls when losing.
**Root cause:** Missing turn counter, no relative advantage features, no graveyard contents, battle not masked when no attackers.
**Fix:** Added turn counter, 3 relative features, graveyard card IDs (20 slots), battle masking. Obs: 349→353, card IDs: 30→50.

## Session 13 — Feb 26 (Network Scaling + Eval Improvements)

### Issue 59: Eval noise masking true performance (50 episodes)
**Problem:** 50-episode evals gave ±6% noise. Agent hit 92% once but couldn't reliably demonstrate 75%+ for self-play activation.
**Fix:** Increased eval episodes to 200 (±3% noise).

### Issue 60: Network bottleneck (4,225 → 256 compression)
**Problem:** 4,225-dimensional input compressed to 256-wide hidden layer. Too aggressive for the information content.
**Fix:** Doubled to 512-wide hidden, policy/value heads [512, 256].

### Issue 61: 5M heuristic-only plateau (~70-77%)
**Problem:** Agent plateaued at 70-77% vs heuristic over 5M steps. Peaked at 80% but never hit 3 consecutive 75%+ evals to trigger self-play.
**Trend:** Win rate oscillated between 60-80% with no upward trend after 1M steps.

### Issue 62: EDOPro bot missing action_cards observation
**Problem:** Bot client only sent `features` and `card_ids` to model, missing the `action_cards` key. Every decision threw KeyError and fell back to first valid action (essentially playing as heuristic).
**Fix:** Added `encode_action_cards(msg, self.card_index)` to bot's observation dict.

## Session 14 — Feb 26-27 (Self-Play + Frozen Pool Training)

### Issue 63: Per-decision opponent mixing (design flaw)
**Problem:** Mixed opponent mode rolled heuristic/recent/older checkpoint per individual action, not per episode. Same game could have different "players" making decisions, creating incoherent opponent behavior.
**Fix:** Changed to per-episode opponent roll in `reset()`. Each game faces a single consistent opponent.

### Issue 64: Frozen checkpoint losing to past versions (25-36%)
**Problem:** During 20M self-play run, current agent consistently lost to its own past checkpoints (25-36% win rate). Flat trend — no improvement over 3M+ steps of self-play.
**Possible causes:** Non-transitivity (rock-paper-scissors dynamics), training signal spread too thin across mixed opponents, deck matchup noise contaminating signal.

### Issue 65: Curriculum stuck at stage 0 (Goat Control only)
**Problem:** Curriculum advancement used vs-checkpoint win rate, which oscillated wildly (25-73%). Never met plateau detection criteria. After 3M+ steps of self-play, still on stage 0.
**Impact:** Agent and all opponents only ever played Goat Control mirrors.

### Issue 66: Mirror vs diverse deck strategy mismatch
**Problem:** All training (heuristic + checkpoints) used the same single-deck pool. Agent never learned to handle different deck strategies or card interactions. Heuristic with diverse decks would teach broader game knowledge.
**Fix:** Separated deck selection by opponent type:
- Heuristic games → diverse decks from full pool (7 decks)
- Checkpoint games → mirror matches (same deck as agent)
Removed curriculum system. All decks available from start.

## Session 15 — Feb 27 (Two-Stream Architecture + Pretrained Embeddings + Attention)

### Issue 67: Single-stream MLP drowns features in embeddings
**Problem:** The extractor concatenated 353 continuous features with 3,872 embedding dims into a 4,225-dim vector, then compressed through a single MLP. Features represented only 8% of input — embeddings dominated, drowning out critical game-state signals like LP, board state, and phase.
**Fix:** Two-stream architecture. Stream A passes features through directly. Stream B compresses embeddings separately. Merged before final MLP, giving features ~66% weight (503/759) instead of 8%.

### Issue 68: Card embeddings randomly initialized
**Problem:** `nn.Embedding` initialized randomly — the agent had to learn card identities from scratch via reward signal alone. With 14k+ cards, this is an enormous search space.
**Fix:** Pre-initialize embeddings from card text using sentence-transformers (`all-MiniLM-L6-v2`, 384-dim → PCA to 32-dim). Cards with similar effects start with similar embeddings, giving the network a semantic head start. Embeddings remain trainable (fine-tuned during training).

### Issue 69: Missing card subtype features in observation
**Problem:** Continuous features encoded basic type (monster/spell/trap) but missed critical subtypes. Agent couldn't distinguish Quick-Play spells (activatable on opponent's turn) from Normal spells, or FLIP effects from regular effects, without relying solely on the embedding.
**Fix:** Added 7 subtype flags across all card slot encoders:
- Monster slots (12→16): +EFFECT, FLIP, FUSION, RITUAL
- Spell/trap slots (6→10): +QUICKPLAY, FIELD, COUNTER, RITUAL
- Hand cards (8→15): +EFFECT, FLIP, FUSION, RITUAL, QUICKPLAY, FIELD, COUNTER
- OBSERVATION_DIM: 353 → 503

### Issue 70: Mean-pooling bottleneck in transformer stream
**Problem:** Initial transformer implementation mean-pooled 121 card slots (50 board + 71 action) down to a single 32-dim vector. Compressing the entire game state (both fields, hands, graveyards) into 32 numbers threw away massive amounts of information.
**Fix:** Replace mean pool with flatten (121×32 = 3,872 dims) → Linear projection to 256. The transformer learns card-to-card interactions, then the projection preserves a much richer representation.

### Issue 71: No card-to-card interaction modeling (architectural limitation)
**Problem:** The MLP-based embedding stream flattened all card slots into one vector and compressed blindly. It couldn't learn relationships like "I have MST in hand AND opponent has a face-down S/T" or "tribute this monster to summon that one."
**Fix:** Replaced linear compression with a 2-layer TransformerEncoder (4 attention heads) over the 121 card slots. Added learned segment embeddings so attention distinguishes board cards from action cards. Cards now attend to each other before the representation is projected.

## Session 16 — Mar 3-5 (Rich Action Encoding + Three-Segment Transformer)

### Issue 72: Actions lack semantic information (plateau at ~72%)
**Problem:** Each action slot was just a card ID — the agent didn't know *what type* of action it is (summon vs activate vs attack), *where* the card is (hand, field, grave), or the card's *stats* (ATK, DEF). The agent plateaued at ~72% vs heuristic for millions of steps with the old architecture.
**Fix:** Replaced flat `action_cards` (71,) with rich `action_features` (71, 12) encoding card ID, 7 action type one-hots, ATK/DEF/level, and card location per action slot.

### Issue 73: Action history too sparse (no card identity)
**Problem:** Old action history (10 actions × 5 features in flat observation) had no card IDs — just player + 4 action type flags. The agent couldn't distinguish which card was summoned or activated, making the history nearly useless for strategic reasoning.
**Fix:** Added structured `action_history` (16, 10) with card embedding IDs, player, action types, ATK/DEF stats, and temporal distance (turns_ago). Removed old 50-dim history from flat features (503 → 453).

### Issue 74: Flatten+project bottleneck (~992K params, rigid)
**Problem:** The two-stream extractor flattened 121 transformer outputs into a fixed 3,872-dim vector, then projected to 256 via a single linear layer. This was parameter-heavy (~992K params for one layer) and couldn't handle variable-length sequences well — padding positions contributed noise despite masking.
**Fix:** Three-segment transformer with segment-aware masked mean pooling. Board (50), action (71), and history (16) tokens each get their own projection to d_model=64, share a TransformerEncoder (4 heads, 2 layers), then pool separately per segment. Pooled segments concatenated (192) → Linear(192, 256). ~49K params for pooling vs ~992K.

**Result:** 86% win rate vs heuristic at 5M steps (old architecture: 72% plateau at 8.8M steps). Floor lifted from ~70% to ~78%.

## Session 17 — Mar 9 (Cross-Attention Action Head)

### Issue 75: Positional action head bottleneck (plateau at ~80%)
**Problem:** Agent plateaued at ~80% vs heuristic. The action head was a simple `Linear(256, 71)` producing logits positionally — it had no awareness of individual action features. The transformer already cross-attended action tokens with board/history tokens, but per-action representations were **pooled away** into a single 64-dim vector before reaching the action head. The model had to memorize which logit position corresponds to which action type, rather than scoring each action based on its content.

**Fix:** Cross-attention action head (`CrossAttentionActionHead`). Instead of pooling action tokens away, store per-action transformer outputs (B, 71, 64) as a side output from the extractor. The new action head concatenates each action's contextual token (64-dim) with the expanded LSTM latent (256-dim) and scores via `Linear(320, 64) → ReLU → Linear(64, 1)`. Each action logit is now conditioned on: (1) what the action is (card stats, action type, location), (2) board context (via transformer self-attention), (3) temporal game state (via LSTM hidden state).

**Parameter count:** Old `Linear(256, 71)` = 18,176 params. New `CrossAttentionActionHead` = 20,608 params. Nearly identical.

**Checkpoint compatibility:** Old checkpoints incompatible — action_net shape changed. Training restarted from scratch.

### Issue 76: Self-play erosion — strategy cycling and meta-overfitting
**Problem:** With cross-attention head, agent reached 76% vs heuristic by 1M steps, then self-play activated (70% gate). Over the next 4M steps, heuristic WR drifted from ~68% average down to ~60%. Classic MARL failure: agent shifted weights to exploit specific habits of its past checkpoints, catastrophically forgetting fundamental strategy needed to beat the aggressive heuristic. The 40/30/30 opponent mix (40% heuristic, 30% recent, 30% older) gave self-play gradients 60% weight — enough to overpower the heuristic anchor signal.

**Fix:** Two changes:
1. **Opponent mix 60/20/20**: 60% heuristic (diverse deck), 20% recent checkpoint (mirror), 20% older checkpoint (mirror). Agent spends the vast majority of training fighting the aggressive baseline.
2. **Self-play gate raised to 85%**: Force the agent to hit its absolute ceiling against the heuristic before allowing self-play. Prevents premature activation where self-play erodes a still-developing foundation.

## Session 18 — Mar 11 (PBRS Reward Shaping + Sequential Multi-Card Selection)

### Issue 77: Sparse terminal reward — zero gradient for intermediate decisions
**Problem:** Agent plateaued at ~70-80% vs heuristic. With sparse terminal-only reward, the agent received zero gradient signal for 50-300 intermediate steps per game. Could not learn trap timing, tribute decisions, or board control — only whether the final outcome was win/loss.
**Fix:** Added potential-based reward shaping (PBRS, Ng et al. 1999). Computes `Φ(s) = 0.4×lp_adv + 0.35×board_adv + 0.25×card_adv` from public game state. Each step receives `shaping_scale × (γ·Φ(s') − Φ(s))` as intermediate reward. Mathematically guaranteed not to change the optimal policy. Configurable via `--shaping-scale` (default 0.5).
**Result:** Agent peaked at 85% vs heuristic at 2.9M steps (previous best: 76%).

### Issue 78: Broken multi-card selection — auto-fill from single index
**Problem:** When the engine asked for 2+ cards (tributes, Graceful Charity discards), `ActionSpace.decode()` auto-filled sequentially from the chosen index: `indices = [(card_idx + i) % len(cards)]`. The agent could never choose optimal card combinations — it always got consecutive cards starting from its pick.
**Fix:** Sequential multi-card selection state machine in `GoatEnv`. When `min_count > 1`, the agent picks one card per step. The action mask excludes already-selected cards. All picks are collected, then sent to the engine as a batch. Opponents still use the old auto-fill path.

## Session 19 — Mar 12 (8 Architecture Improvements)

Agent peaked at 85% vs heuristic with PBRS but oscillated in the 60-72% range post self-play. Identified 8 architectural weaknesses, split into Phase A (backward-compatible) and Phase B (checkpoint-breaking, retrain from scratch).

### Improvement A1: Engine State Hash Check

**Problem:** Issues #10, #40, #42 were silent struct bugs that corrupted training for 100k+ steps. No validation existed to catch state desync between Python GameState and the C++ engine.

**Fix:** Added `Duel.verify_state()` method that queries the engine via `query_card()` for each MZONE/SZONE slot and compares card codes against Python GameState. Added debug hooks in `GoatEnv.reset()` and `GoatEnv.step()` via `if __debug__:` — zero cost in production (`python -O`), immediate feedback during development. New `test_state_sync.py` runs 10 heuristic games with verification after every step.

### Improvement A2: Action Space Overflow Logging

**Problem:** When >5 summonable cards (or >10 activatable) are available, the excess is silently dropped by list slicing. The agent never knows options were lost.

**Fix:** Added `logging.getLogger` warnings on first overflow per action type (summon, spsummon, set_monster, set_st, activate, reposition) in `ActionSpace.get_mask()`. Logs count and how many dropped. Already caught overflows in the first training run: `set_st` and `summon` with 6 options capped at 5.

### Improvement A3: PBRS Hybrid Potential Function

**Problem:** The old potential function used count-based `board_adv = (my_monsters - opp_monsters) / 5`. Tributing 2 weak monsters for 1 strong one registered as -1 card advantage. But fully replacing card count with ATK sum would create a blind spot for Spells/Traps (Pot of Greed = +1 card but 0 ATK difference).

**Fix:** Hybrid potential: `Φ(s) = 0.40 × atk_adv + 0.25 × card_adv + 0.35 × lp_adv`. When card DB is available, `atk_adv` sums ATK/5000 for face-up monsters per side (clamped to [-1,1]). Falls back to count-based when DB is None. `card_adv` kept so drawing spells and setting traps still register. Passed `db=self._card_db` to both `compute_potential()` call sites in GoatEnv.

### Improvement A4: Face-Down Card Age Tracking

**Problem:** Prerequisite for B2. No way to represent how long a face-down card has been set — a key signal for experienced players (a card set for 5 turns is more likely a trap than one set this turn).

**Fix:** Added `set_turn: int = 0` to `ZoneCard`. In `_handle_move()`, destination cards get `set_turn=state.current_turn`. Used by B2's opponent inference features.

### Improvement B1: Richer Action History (16×10 → 16×13)

**Problem:** History had no outcomes — "opponent activated Mirror Force" but not "...destroyed 3 monsters." The agent couldn't learn that certain activations are dangerous.

**Fix:** Added `_pending_attack_record` to GameState for damage attribution. On MsgDamage following an attack, annotates `damage` on that specific record (prevents mis-attributing burn damage). On MsgMove (field→grave), increments `destroyed` on the last activate record. On MsgChainNegated, marks `was_negated`. Three new history columns: `damage/8000` (col 10), `cards_destroyed/5` (col 11), `was_negated` (col 12). Also tracks `extra_draws` per player for B2.

### Improvement B2: Opponent Hand Inference + Face-Down Age (453 → 462)

**Problem:** Agent had zero representation of what opponent might hold, how old set cards are, or graveyard composition.

**Fix:** 9 new features appended to observation:
- `opp_extra_draws / 5.0` — draws beyond normal draw phase (Pot of Greed indicator)
- `opp_facedown_count / 5.0` — total face-down cards
- Per opponent S/T slot × 5: `turns_since_set / 10.0` (uses `ZoneCard.set_turn` from A4)
- `opp_graveyard_monster_count / 10.0` — GY monster depth (Chaos fuel indicator)
- `opp_banished_count / 10.0` — Chaos plays used

OBSERVATION_DIM: 453 → 462. Merge MLP adjusts automatically.

### Improvement B3: Board Token Preservation (cross-attention value head)

**Problem:** 50 board tokens were mean-pooled to a single 64-dim vector. The critic couldn't assess specific board positions (e.g. "3000 ATK threat in Zone 2 is lethal").

**Fix:** Extractor now stores `_last_board_tokens` (50×64, padded positions zeroed) alongside `_last_action_tokens`. New `BoardAttentionValueHead` replaces `Linear(256, 1)`: projects latent_vf (256) to a query (64), attends over 50 board tokens (single-head cross-attention with padding mask), concatenates attended context with latent_vf, projects to scalar V(s). The critic path now has per-card board awareness while the actor path remains unchanged.

### Improvement B4: Increase Card Embedding Dimension (32 → 64)

**Problem:** 14k cards in 32 dimensions is tight for distinguishing strategically different cards. PCA at 32d captured only ~47% of variance from the sentence-transformer encodings.

**Fix:** Doubled `embed_dim` to 64. Since 64 matches `d_model`, `board_project` becomes `nn.Identity()` (saves parameters and computation — no projection needed). PCA at 64d captures 64.4% variance. Modern GPU tensor cores process 64-wide blocks optimally, so no speed penalty vs 48d. Regenerated `data/card_embeddings.npy` at (14337, 64).

**Checkpoint compatibility:** All Phase B changes break checkpoints. Training restarted from scratch with the full v3 architecture.

## Session 20 — Mar 12-15 (B3 Ablation + Hyperparameter Tuning)

### Issue 79: BoardAttentionValueHead causing critic failure (explained variance declining)

**Problem:** With all 8 Phase B improvements live, training plateaued at 65% vs heuristic for 8M+ steps. Explained variance *declined* from 0.44 → 0.34 over training — the critic was getting worse, not better. The old architecture reached 85% with the same hyperparameters.

**Diagnosis:** B3 (cross-attention value head) was the prime suspect. Adding attention to the critic while keeping the actor unchanged made the critic much harder to optimize. The hyperparameters (batch_size=128, n_epochs=2) were tuned for the simpler Linear(256,1) critic, not an attention-based one.

**Fix:** Added `--no-board-attention` toggle to disable B3 and fall back to `Linear(256, 1)` value head. Cannot resume from old checkpoint (weight shape mismatch) — restarted training from scratch.

### Issue 80: Hyperparameters sub-optimal for larger architecture

**Problem:** Phase B's larger model (64d embeddings, richer history, attention heads) needed different hyperparameters than the original architecture. Low batch size (128) caused high gradient noise; few epochs (2) gave insufficient critic updates; high entropy coef (0.10) prevented the agent from committing to learned strategies.

**Fix:** Tuned hyperparameters:
- `batch_size`: 128 → 256 (stabler gradients)
- `n_epochs`: 2 → 4 (more critic updates per rollout)
- `ent_coef`: 0.10 → 0.05 (less forced exploration)
- `vf_coef`: 0.5 → 0.8 (stronger critic gradient signal)
- Added `--ent-coef`, `--vf-coef` CLI args and injection on `--resume`

**Result:** With B3 disabled + tuned hyperparams, agent hit 88% by 6M steps, 90% peak by 15M steps. Explained variance climbed to 0.51 (vs 0.34 with B3 on). Training stable in the 82-90% range.

### Issue 81: SelfPlayCallback loses checkpoint pool on resume

**Problem:** When resuming from a checkpoint, the `SelfPlayCallback` started with an empty pool and `_self_play_active=False`. Self-play had to re-activate from scratch, losing all progress.

**Fix:** Added checkpoint discovery in `_init_callback()`: scans `save_dir` for existing `ckpt_*.zip` files, populates the pool, fast-forwards `_next_checkpoint_step`, and re-activates self-play if ≥2 checkpoints exist.

### Issue 82: Parallel envs speedup — 8 → 16 envs

**Problem:** Training at 80-100 FPS with 8 envs. GPU at 26% utilization, CPU at 30%. Neither saturated.

**Fix:** Bumped `--n-envs` to 16. FPS increased to 108-157 (60-90% faster). CPU now at ~80%, GPU still has headroom.

## Session 21 — Mar 16-20 (Human Testing + Phase C: Structured Card Semantics)

### Issue 83: Agent plays randomly against human — no card understanding

**Problem:** Connected trained agent (86% vs heuristic) to EDOPro for human testing. Agent exhibited fundamental misplays:
- MST-ing its own backrow
- Attacking with monsters that have lower ATK than opponent's
- Book of Moon on its own monsters
- Activating spells for no strategic reason
- Setting cards pointlessly
- Using card effects against itself

85-90% vs heuristic is meaningless — the heuristic is so weak that random-ish play wins most games. The agent had no understanding of what its actions actually do.

**Root cause:** Three observation gaps:
1. **No target ownership context** — during SELECT_CARD (target for MST, Book of Moon), the observation didn't indicate whether each candidate card belonged to the agent or opponent
2. **No card effect semantics** — the 64d embedding captured text similarity but zero gameplay meaning. No feature said "MST destroys spells/traps" or "Book of Moon flips face-down"
3. **No ATK comparison at decision time** — the per-action tokens for attack actions didn't include opponent's monster ATK for comparison

### Issue 84: EDOPro card selection response format (RETRY errors)

**Problem:** Bot got MSG_RETRY when responding to SELECT_CARD in EDOPro. The local engine uses `set_responseb` with byte format (count + byte indices), but EDOPro's ygopro-core fork (edo9300) uses `parse_response_cards` which expects int32 format: `type(int32) + count(uint32) + indices(uint32 each)`.

**Fix:** `DuelProxy.respond_card_selection` changed to int32 format: `struct.pack("<i", 0)` (type=0) + `struct.pack("<I", count)` + `struct.pack("<I", idx)` per index.

### Issue 85: SORT_CARD unhandled — causes RETRY and disconnect

**Problem:** EDOPro sent SORT_CARD (MSG 25) which the bot tried to handle with the model, producing invalid responses.

**Fix:** Added explicit handler in bot's `_respond_to_decision`: if msg_type is SORT_CARD or SORT_CHAIN, respond with `-1` (auto-sort) instead of using the model.

### Phase C Implementation: Structured Card Semantics (action_features 12 → 29)

Three targeted upgrades to the action feature encoding, all touching `observation.py`:

**C1. Per-card ownership flags (cols 12-13):**
During SELECT_CARD, SELECT_TRIBUTE, and SELECT_CHAIN, each candidate card gets `target_is_mine` (col 12) and `target_is_opp` (col 13) flags based on `CardInfo.controller == perspective`. Stops 70% of self-MST/self-Book-of-Moon mistakes because the policy now sees "I'm about to destroy my own card."

**C2. Effect category flags (cols 14-25):**
New `effect_flags.py` lookup table mapping ~137 GOAT format card codes to 12 binary flags: destroys_monster, destroys_spelltrap, negates, draws_cards, searches_deck, flips_facedown, burns_lp, bounces_to_hand, special_summons, gains_atk, changes_position, protects. Appended to every action slot via `_encode_action_slot`. Unknown cards get all zeros (safe default).

**C3. ATK matchup features (cols 26-28) for attack actions:**
When building action features for `MsgSelectBattleCmd` attackable slots: computes strongest face-up opponent monster ATK from GameState, and whether opponent has face-down monsters. Each attacker gets: `target_atk_norm` (max opp ATK / 5000), `atk_advantage` ((my ATK - max opp ATK) / 5000, clamped [-1,1]), `opp_has_facedown`. The cross-attention action head can now learn "don't attack 1200 into 2400."

**C4. Target-aware action head:**
Effectively implemented by C1-C3 — the cross-attention action head already scores each action using its rich per-action token (now 29 features including ownership, effect type, and ATK matchup) combined with the LSTM latent. No additional architectural change needed.

`encode_action_features` now accepts `perspective` and `state` parameters. All call sites updated (goat_env.py ×2, bot.py ×1). ACTION_FEATURES_DIM: 12 → 29. Extractor auto-adapts (dynamic shape). Checkpoints incompatible — training restarted from scratch.

---

## Critical Bugs Summary (by impact)

1. **CardData struct setcode field** (#40) — 16-byte offset mismatch. Engine saw type=0 for all cards. Agent could never summon. All prior training broken.
2. **TAG_MODE accidentally enabled** (#10) — 0x20 flag caused instant deck-out wins. False 100% win rate.
3. **Battle command categories swapped** (#42) — Attacks decoded as activations. Hangs + inability to attack.
4. **5 competing reward signals** (#51) — Agent couldn't learn from contradictory shaping signals.
5. **Self-play co-adaptation collapse** (#54, #57) — Current-self opponent caused policy degradation.
6. **Heuristic attack-cancel loop** (#46) — Chain-pass cancelled attack targets, infinite battle loop.
7. **No action-to-card mapping** (#33) — Agent couldn't connect actions to cards on field/hand.
8. **Observation blind spots** (#58) — Missing turn count, relative advantage, graveyard contents.
9. **Per-decision opponent mixing** (#63) — Opponent type changed mid-game, incoherent opponent behavior.
10. **Single-deck training** (#65, #66) — Curriculum stuck at Goat Control only. Agent never exposed to diverse strategies.
11. **BoardAttentionValueHead critic failure** (#79) — Cross-attention value head too hard to optimize. Explained variance declined during training. Agent plateaued at 65%.
12. **No card understanding at decision time** (#83) — Agent couldn't distinguish own vs opponent cards during targeting, didn't know card effects, couldn't compare ATK values. 86% vs heuristic but played randomly vs humans.
