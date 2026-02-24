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
from .observation import encode_observation, encode_card_ids, CardDB, OBSERVATION_DIM, CARD_ID_DIM
from .card_index import CardIndex
from .action_space import ActionSpace, ACTION_DIM, _respond_select_place
from .reward import compute_reward, card_count
from .heuristic import heuristic_action


# ============================================================
# Default paths (relative to project root)
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LIB = _PROJECT_ROOT / "lib" / "libocgcore.so"
_DEFAULT_DB = _PROJECT_ROOT / "data" / "cards.cdb"
_DEFAULT_SCRIPT_DIR = _PROJECT_ROOT / "data" / "script"
_DEFAULT_DECK = _PROJECT_ROOT / "data" / "deck" / "Goat Control.ydk"

# Max steps per episode before truncation.
# Active GOAT games finish well within 200 steps.
# Truncation penalty (-0.5) teaches the agent not to stall.
_MAX_STEPS = 200


class GoatEnv(gym.Env):
    """
    Gymnasium environment for GOAT-format Yu-Gi-Oh! duels.

    Observation space: Dict(features=Box(329,), card_ids=Box(30,))
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
        step_penalty: float = 0.0,
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
        self._step_penalty = step_penalty

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
                low=0.0, high=1.0,
                shape=(OBSERVATION_DIM,),
                dtype=np.float32,
            ),
            "card_ids": spaces.Box(
                low=0.0,
                high=float(self._card_index.vocab_size - 1),
                shape=(CARD_ID_DIM,),
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
        self._prev_lp: Optional[tuple[int, int]] = None
        self._prev_cards: Optional[tuple[int, int]] = None

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

        # Pick opponent deck (weighted random from pool if available)
        opp_deck = self._opp_deck
        if self._opp_deck_pool:
            if self._opp_deck_weights is not None:
                idx = int(np.random.choice(len(self._opp_deck_pool), p=self._opp_deck_weights))
                opp_deck = self._opp_deck_pool[idx]
            else:
                opp_deck = random.choice(self._opp_deck_pool)

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
        self._prev_lp = None
        self._prev_cards = None

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

        # Snapshot LP and card counts before advancing for reward shaping
        state = self._duel.state
        prev_lp = (
            state.players[self._agent_player].lp,
            state.players[1 - self._agent_player].lp,
        )
        prev_cards = (
            card_count(state, self._agent_player),
            card_count(state, 1 - self._agent_player),
        )

        # If there's a pending decision, send the agent's action
        if self._pending_msg is not None:
            self._action_space_handler.decode(action, self._pending_msg, self._duel)

        # Advance until next agent decision or game end
        self._pending_msg = self._advance()

        state = self._duel.state
        terminated = state.is_finished
        truncated = self._step_count >= _MAX_STEPS and not terminated

        reward = compute_reward(
            state, self._agent_player,
            prev_lp=prev_lp,
            prev_cards=prev_cards,
            step_penalty=self._step_penalty,
        )

        # Truncation penalty: running out the clock is bad
        if truncated:
            reward = -0.5

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def valid_action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions for the current pending message."""
        if self._pending_msg is None:
            # No pending decision — all actions valid (shouldn't happen in practice)
            return np.ones(ACTION_DIM, dtype=bool)
        return self._action_space_handler.get_mask(self._pending_msg)

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

    # ============================================================
    # Internal helpers
    # ============================================================
    # Max opponent decisions per _advance() call before force-ending.
    # Prevents infinite loops from heuristic/opponent policies.
    _MAX_OPP_DECISIONS = 500

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
                ),
                "card_ids": encode_card_ids(
                    duel.state,
                    perspective=opp_perspective,
                    card_index=self._card_index,
                ),
            }
            action = self._opponent_fn(opp_obs, mask)
            # Clamp to valid actions
            if not mask[action]:
                action = int(np.random.choice(valid))
        else:
            action = heuristic_action(mask, is_chain=isinstance(msg, MsgSelectChain))

        self._action_space_handler.decode(action, msg, duel)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Encode current game state as observation dict."""
        if self._duel is None:
            return {
                "features": np.zeros(OBSERVATION_DIM, dtype=np.float32),
                "card_ids": np.zeros(CARD_ID_DIM, dtype=np.float32),
            }
        return {
            "features": encode_observation(
                self._duel.state,
                perspective=self._agent_player,
                db=self._card_db,
            ),
            "card_ids": encode_card_ids(
                self._duel.state,
                perspective=self._agent_player,
                card_index=self._card_index,
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
