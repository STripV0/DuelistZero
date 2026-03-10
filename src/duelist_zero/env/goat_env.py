"""
GoatEnv — Gymnasium environment for GOAT-format Yu-Gi-Oh! duels.

The agent plays as player 0 or 1 (randomized each episode).
The default opponent uses a heuristic policy (summon, attack, etc.).

Usage:
    env = GoatEnv()
    obs, info = env.reset()
    mask = env.valid_action_mask()
    action = np.random.choice(np.where(mask)[0])
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()

Compatible with stable-baselines3 MaskablePPO via valid_action_mask().
"""

import random
import struct
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..core.bindings import OcgCore
from ..core.callbacks import CallbackManager
from ..core.constants import (
    GOAT_DUEL_OPTIONS,
    LOCATION,
    POSITION,
    MSG,
    PROCESSOR_END,
    PROCESSOR_WAITING,
)
from ..core.message_parser import (
    MessageParser,
    ParsedMessage,
    MsgSelectIdleCmd,
    MsgSelectBattleCmd,
    MsgSelectCard,
    MsgSelectChain,
    MsgSelectEffectYn,
    MsgSelectYesNo,
    MsgSelectOption,
    MsgSelectPosition,
    MsgSelectPlace,
    MsgSelectTribute,
    MsgWin,
    RESPONSE_MESSAGES,
)
from ..engine.duel import Duel, load_deck
from ..engine.game_state import GameState
from .observation import (
    encode_observation, encode_card_ids, encode_action_features, encode_action_history,
    CardDB, OBSERVATION_DIM, CARD_ID_DIM,
    ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM,
    HISTORY_LENGTH, HISTORY_FEATURES_DIM,
)
from .card_index import CardIndex
from .action_space import ActionSpace, ACTION_DIM, _respond_select_place
from .reward import compute_reward
from .heuristic import heuristic_action


class _RecurrentOpponentFn:
    """Stateful wrapper for recurrent (LSTM) model opponents.

    Tracks LSTM hidden state across decisions within an episode.
    Call reset() at the start of each new episode.
    """

    def __init__(self, model, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic
        self.state = None
        self.episode_start = np.array([True])

    def reset(self):
        self.state = None
        self.episode_start = np.array([True])

    def __call__(self, obs, mask):
        action, self.state = self.model.predict(
            obs,
            state=self.state,
            episode_start=self.episode_start,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        self.episode_start = np.array([False])
        return int(action)


# ============================================================
# Default paths (relative to project root)
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LIB = _PROJECT_ROOT / "lib" / "libocgcore.so"
_DEFAULT_DB = _PROJECT_ROOT / "data" / "cards.cdb"
_DEFAULT_SCRIPT_DIR = _PROJECT_ROOT / "data" / "script"
_DEFAULT_DECK = _PROJECT_ROOT / "data" / "deck" / "Goat Control.ydk"

# Max steps per episode before truncation.
# Truncation penalty (-1.0) treats stalling as equivalent to a loss.
_MAX_STEPS = 200


class GoatEnv(gym.Env):
    """
    Gymnasium environment for GOAT-format Yu-Gi-Oh! duels.

    Observation space: Dict(features=Box(349,), card_ids=Box(30,))
    Action space: Discrete(ACTION_DIM)

    The agent plays as player 0 or 1 (randomized each episode).
    Default opponent uses a heuristic policy.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        lib_path: str | Path = _DEFAULT_LIB,
        db_path: str | Path = _DEFAULT_DB,
        script_dir: str | Path = _DEFAULT_SCRIPT_DIR,
        deck_path: str | Path = _DEFAULT_DECK,
        opponent_deck_path: Optional[str | Path] = None,
        opponent_deck_pool: Optional[list[str | Path]] = None,
        opponent_deck_weights: Optional[list[float]] = None,
        opponent_fn: Optional[Callable[[np.ndarray, np.ndarray], int]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.lib_path = Path(lib_path)
        self.db_path = Path(db_path)
        self.script_dir = Path(script_dir)
        self.deck_path = Path(deck_path)
        self.opponent_deck_path = Path(opponent_deck_path) if opponent_deck_path else self.deck_path
        self.render_mode = render_mode
        self._opponent_fn = opponent_fn

        # Load engine (shared across episodes)
        self._core = OcgCore(self.lib_path)
        self._cb = CallbackManager(self._core, self.db_path, self.script_dir)
        self._cb.register()

        # Card DB for observation encoding
        self._card_db = CardDB(self.db_path)

        # Card index for embedding lookups
        self._card_index = CardIndex(self.db_path)

        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-1.0, high=1.0,
                shape=(OBSERVATION_DIM,),
                dtype=np.float32,
            ),
            "card_ids": spaces.Box(
                low=0.0,
                high=float(self._card_index.vocab_size - 1),
                shape=(CARD_ID_DIM,),
                dtype=np.float32,
            ),
            "action_features": spaces.Box(
                low=0.0,
                high=float(self._card_index.vocab_size - 1),
                shape=(ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM),
                dtype=np.float32,
            ),
            "action_history": spaces.Box(
                low=0.0,
                high=float(self._card_index.vocab_size - 1),
                shape=(HISTORY_LENGTH, HISTORY_FEATURES_DIM),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(ACTION_DIM)

        # Action space handler
        self._action_space_handler = ActionSpace()

        # Load decks
        self._agent_deck = load_deck(self.deck_path)
        self._opp_deck = load_deck(self.opponent_deck_path)

        # Opponent deck pool (loaded lazily, sampled each reset)
        self._opp_deck_pool: Optional[list[tuple]] = None
        self._opp_deck_weights: Optional[list[float]] = None
        if opponent_deck_pool:
            self._opp_deck_pool = [load_deck(Path(p)) for p in opponent_deck_pool]
            if opponent_deck_weights and len(opponent_deck_weights) == len(opponent_deck_pool):
                self._opp_deck_weights = opponent_deck_weights

        # Episode state
        self._duel: Optional[Duel] = None
        self._pending_msg: Optional[ParsedMessage] = None
        self._step_count = 0
        self._agent_player = 0

        # Deck identity indices for observation encoding
        self._agent_deck_idx: int = 0
        self._opp_deck_idx: int = 0

        # Per-episode opponent selection (set via set_opponent_from_path)
        self._opponent_mode: str = "heuristic"
        self._opponent_models: dict = {}

        # Tribute-cancel loop prevention (Fix 2)
        self._cancelled_tribute_code: Optional[int] = None
        self._last_idle_action: Optional[int] = None
        self._last_idle_msg: Optional[MsgSelectIdleCmd] = None

        # Seeding
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ============================================================
    # Gymnasium API
    # ============================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        # Clean up previous duel
        if self._duel is not None:
            self._duel.end()
            self._duel = None

        # Randomize which player the agent controls each episode
        self._agent_player = random.choice([0, 1])

        # Per-episode opponent and deck selection
        if self._opponent_mode == "mixed":
            # Roll opponent type for this episode
            r = np.random.random()
            if r < 0.60:
                # Heuristic with diverse deck from pool
                self._opponent_fn = None
                opp_deck = self._pick_diverse_deck()
            elif r < 0.80:
                # Recent frozen checkpoint with mirror deck
                opp = self._opponent_models.get("recent")
                if opp is not None:
                    opp.reset()  # Reset LSTM state for new episode
                    self._opponent_fn = opp
                else:
                    self._opponent_fn = None
                opp_deck = self._agent_deck  # mirror
                self._opp_deck_idx = self._agent_deck_idx
            else:
                # Older frozen checkpoint with mirror deck
                opp = self._opponent_models.get("older")
                if opp is not None:
                    opp.reset()  # Reset LSTM state for new episode
                    self._opponent_fn = opp
                else:
                    self._opponent_fn = None
                opp_deck = self._agent_deck  # mirror
                self._opp_deck_idx = self._agent_deck_idx
        elif self._opponent_mode == "heuristic":
            # Heuristic with diverse deck from pool
            self._opponent_fn = None
            opp_deck = self._pick_diverse_deck()
        else:
            # "model" mode — mirror deck, reset LSTM state
            if hasattr(self._opponent_fn, "reset"):
                self._opponent_fn.reset()
            opp_deck = self._agent_deck
            self._opp_deck_idx = self._agent_deck_idx

        # Create new duel — assign decks based on player order
        # Player 0 = deck1, Player 1 = deck2
        if self._agent_player == 0:
            deck1, deck2 = self._agent_deck, opp_deck
        else:
            deck1, deck2 = opp_deck, self._agent_deck

        duel_seed = random.randint(0, 0xFFFFFFFF)
        self._duel = Duel(self._core, self._cb, seed=duel_seed)
        self._duel._deck1 = deck1
        self._duel._deck2 = deck2
        self._duel.start()

        self._step_count = 0
        self._cancelled_tribute_code = None
        self._last_idle_action = None
        self._last_idle_msg = None

        # Process until agent's first decision.
        # If the game ends before the agent gets a turn (e.g. opponent
        # wins immediately), re-create the duel to avoid a stuck state.
        for _ in range(10):
            self._pending_msg = self._advance()
            if self._pending_msg is not None:
                break
            # Game ended before agent's turn — restart
            self._duel.end()
            duel_seed = random.randint(0, 0xFFFFFFFF)
            self._duel = Duel(self._core, self._cb, seed=duel_seed)
            self._duel._deck1 = deck1
            self._duel._deck2 = deck2
            self._duel.start()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        assert self._duel is not None, "Call reset() before step()"

        self._step_count += 1

        # If there's a pending decision, send the agent's action
        if self._pending_msg is not None:
            # Track idle cmd context for tribute-cancel detection
            if isinstance(self._pending_msg, MsgSelectIdleCmd):
                self._last_idle_action = action
                self._last_idle_msg = self._pending_msg

            # Detect tribute cancel → record which card code to block
            if isinstance(self._pending_msg, MsgSelectTribute) and action == 50:
                # action 50 = _YESNO_NO = cancel
                if self._last_idle_msg is not None and self._last_idle_action is not None:
                    code = self._resolve_idle_card_code(
                        self._last_idle_action, self._last_idle_msg
                    )
                    if code is not None:
                        self._cancelled_tribute_code = code
            else:
                # Any non-cancel action clears the block
                self._cancelled_tribute_code = None

            self._action_space_handler.decode(action, self._pending_msg, self._duel)

        # Advance until next agent decision or game end
        self._pending_msg = self._advance()

        state = self._duel.state
        terminated = state.is_finished
        truncated = self._step_count >= _MAX_STEPS and not terminated

        reward = compute_reward(state, self._agent_player)

        # Truncation penalty: stalling is as bad as losing
        if truncated:
            reward = -1.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def valid_action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions for the current pending message."""
        if self._pending_msg is None:
            # No pending decision — all actions valid (shouldn't happen in practice)
            return np.ones(ACTION_DIM, dtype=bool)
        mask = self._action_space_handler.get_mask(self._pending_msg)

        # Mask "go to battle" when no attack-position monsters on field
        if isinstance(self._pending_msg, MsgSelectIdleCmd) and mask[35]:
            me = self._duel.state.players[self._agent_player]
            has_attacker = any(
                m is not None and bool(m.position & POSITION.ATTACK)
                for m in me.monsters
            )
            if not has_attacker:
                mask[35] = False
                # Ensure at least one action remains
                if not mask.any():
                    mask[35] = True

        # Fix 2: Block summon/set of a card whose tribute was just cancelled
        if (
            self._cancelled_tribute_code is not None
            and isinstance(self._pending_msg, MsgSelectIdleCmd)
        ):
            msg = self._pending_msg
            blocked = self._cancelled_tribute_code
            for i, c in enumerate(msg.summonable[:5]):
                if c.code == blocked:
                    mask[0 + i] = False
            for i, c in enumerate(msg.setable_monsters[:5]):
                if c.code == blocked:
                    mask[10 + i] = False
            # Ensure at least one action remains valid
            if not mask.any():
                mask = self._action_space_handler.get_mask(self._pending_msg)
                self._cancelled_tribute_code = None

        return mask

    def render(self) -> Optional[str]:
        if self._duel is None:
            return None
        state = self._duel.state
        lines = []
        lines.append("=" * 50)
        lines.append(f"Turn {state.current_turn} | Player {state.turn_player}'s turn")
        for p in [0, 1]:
            ps = state.players[p]
            tag = "AGENT" if p == self._agent_player else "OPP"
            lines.append(
                f"  [{tag}] LP={ps.lp} Hand={ps.hand_count} "
                f"Deck={ps.deck_count} GY={ps.grave_count}"
            )
            monsters = [f"M{i}:{c.code}" for i, c in enumerate(ps.monsters) if c]
            spells = [f"S{i}:{c.code}" for i, c in enumerate(ps.spells) if c]
            if monsters:
                lines.append(f"    Monsters: {' '.join(monsters)}")
            if spells:
                lines.append(f"    Spells:   {' '.join(spells)}")
        lines.append("=" * 50)
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def close(self):
        if self._duel is not None:
            self._duel.end()
            self._duel = None
        if self._card_db is not None:
            self._card_db.close()
            self._card_db = None

    def set_opponent(self, opponent_fn: Optional[Callable[[np.ndarray, np.ndarray], int]]):
        """
        Set the opponent policy function.

        Args:
            opponent_fn: Callable(obs, mask) -> action_idx, or None for random.
        """
        self._opponent_fn = opponent_fn

    def set_deck_pool(
        self,
        pool: list[str | Path],
        weights: Optional[list[float]] = None,
    ) -> None:
        """
        Update the opponent deck pool and sampling weights.

        Args:
            pool: List of .ydk file paths.
            weights: Optional sampling weights (must sum to ~1.0).
        """
        self._opp_deck_pool = [load_deck(Path(p)) for p in pool]
        if weights and len(weights) == len(pool):
            self._opp_deck_weights = weights
        else:
            self._opp_deck_weights = None

    def set_opponent_from_path(
        self,
        mode: str = "heuristic",
        model_path: Optional[str] = None,
        past_path: Optional[str] = None,
    ) -> None:
        """
        Set the opponent policy using only picklable arguments.

        This method is safe to call via SubprocVecEnv.env_method() since
        it only accepts strings/None (no closures or model objects).

        Modes:
            "heuristic": Heuristic opponent with diverse decks from pool.
            "model": Single checkpoint opponent with mirror deck.
            "mixed": Per-episode roll — 60% heuristic (diverse deck),
                     20% recent frozen (mirror), 20% older frozen (mirror).

        Args:
            mode: One of "heuristic", "model", "mixed".
            model_path: Path to frozen recent checkpoint (for "model"/"mixed").
            past_path: Path to frozen older checkpoint (for "mixed").
        """
        # Explicitly free old opponent models before loading new ones
        import gc
        self._opponent_fn = None
        self._opponent_models = {}
        gc.collect()

        if mode == "heuristic":
            self._opponent_mode = "heuristic"
            return

        from ..training.maskable_recurrent_ppo import MaskableRecurrentPPO

        if mode == "model":
            assert model_path is not None
            opp_model = MaskableRecurrentPPO.load(model_path, device="cpu")
            self._opponent_fn = _RecurrentOpponentFn(opp_model)
            self._opponent_mode = "model"

        elif mode == "mixed":
            # Load models once, wrap with stateful LSTM tracking
            recent = (
                _RecurrentOpponentFn(MaskableRecurrentPPO.load(model_path, device="cpu"))
                if model_path else None
            )
            older = (
                _RecurrentOpponentFn(MaskableRecurrentPPO.load(past_path, device="cpu"))
                if past_path else None
            )
            self._opponent_mode = "mixed"
            self._opponent_models = {
                "recent": recent,
                "older": older,
            }

    # ============================================================
    # Internal helpers
    # ============================================================
    # Max opponent decisions per _advance() call before force-ending.
    # Prevents infinite loops from heuristic/opponent policies.
    _MAX_OPP_DECISIONS = 500

    def _pick_diverse_deck(self) -> tuple:
        """Pick an opponent deck from the pool, or fall back to default."""
        if self._opp_deck_pool:
            if self._opp_deck_weights is not None:
                idx = int(np.random.choice(len(self._opp_deck_pool), p=self._opp_deck_weights))
            else:
                idx = random.randrange(len(self._opp_deck_pool))
            self._opp_deck_idx = idx
            return self._opp_deck_pool[idx]
        self._opp_deck_idx = 0
        return self._opp_deck

    def _advance(self) -> Optional[ParsedMessage]:
        """
        Process the engine until the agent needs to make a decision,
        or the game ends. Opponent decisions are handled automatically.

        Returns the pending decision message for the agent, or None if done.
        """
        duel = self._duel
        opp_decisions = 0

        while not duel.state.is_finished:
            msg = duel.process()

            if msg is None:
                return None

            player = getattr(msg, "player", 0)

            if player == self._agent_player:
                # Fix 1: auto-respond if only one valid action (forced decision)
                mask = self._action_space_handler.get_mask(msg)
                valid = np.where(mask)[0]
                if len(valid) == 1:
                    self._action_space_handler.decode(int(valid[0]), msg, duel)
                    continue
                return msg
            else:
                opp_decisions += 1
                if opp_decisions > self._MAX_OPP_DECISIONS:
                    duel.state.is_finished = True
                    return None
                self._opponent_response(msg, duel)

        return None

    def _opponent_response(self, msg: ParsedMessage, duel: Duel) -> None:
        """Send opponent's response: use opponent_fn if set, else random."""
        mask = self._action_space_handler.get_mask(msg)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            duel.respond_int(0)
            return

        if self._opponent_fn is not None:
            # Compute observation from opponent's perspective
            opp_perspective = 1 - self._agent_player
            opp_obs = {
                "features": encode_observation(
                    duel.state,
                    perspective=opp_perspective,
                    db=self._card_db,
                    deck_id=self._opp_deck_idx,
                ),
                "card_ids": encode_card_ids(
                    duel.state,
                    perspective=opp_perspective,
                    card_index=self._card_index,
                ),
                "action_features": encode_action_features(
                    msg,
                    self._card_index,
                    db=self._card_db,
                ),
                "action_history": encode_action_history(
                    duel.state,
                    perspective=opp_perspective,
                    card_index=self._card_index,
                    db=self._card_db,
                ),
            }
            action = self._opponent_fn(opp_obs, mask)
            # Clamp to valid actions
            if not mask[action]:
                action = int(np.random.choice(valid))
        else:
            action = heuristic_action(mask, is_chain=isinstance(msg, MsgSelectChain))

        self._action_space_handler.decode(action, msg, duel)

    def _resolve_idle_card_code(
        self, action: int, msg: MsgSelectIdleCmd
    ) -> Optional[int]:
        """Return the card code targeted by an idle cmd action, or None."""
        if 0 <= action < 5 and action < len(msg.summonable):
            return msg.summonable[action].code
        if 10 <= action < 15 and (action - 10) < len(msg.setable_monsters):
            return msg.setable_monsters[action - 10].code
        return None

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Encode current game state as observation dict."""
        if self._duel is None:
            return {
                "features": np.zeros(OBSERVATION_DIM, dtype=np.float32),
                "card_ids": np.zeros(CARD_ID_DIM, dtype=np.float32),
                "action_features": np.zeros(
                    (ACTION_FEATURES_SLOTS, ACTION_FEATURES_DIM), dtype=np.float32
                ),
                "action_history": np.zeros(
                    (HISTORY_LENGTH, HISTORY_FEATURES_DIM), dtype=np.float32
                ),
            }
        return {
            "features": encode_observation(
                self._duel.state,
                perspective=self._agent_player,
                db=self._card_db,
                deck_id=self._agent_deck_idx,
            ),
            "card_ids": encode_card_ids(
                self._duel.state,
                perspective=self._agent_player,
                card_index=self._card_index,
            ),
            "action_features": encode_action_features(
                self._pending_msg,
                self._card_index,
                db=self._card_db,
            ),
            "action_history": encode_action_history(
                self._duel.state,
                perspective=self._agent_player,
                card_index=self._card_index,
                db=self._card_db,
            ),
        }

    def _get_info(self) -> dict:
        """Return info dict."""
        if self._duel is None:
            return {}
        state = self._duel.state
        return {
            "turn": state.current_turn,
            "winner": state.winner,
            "step_count": self._step_count,
            "agent_lp": state.players[self._agent_player].lp,
            "opp_lp": state.players[1 - self._agent_player].lp,
        }
