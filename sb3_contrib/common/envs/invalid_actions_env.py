from typing import Optional
from typing import Any, Generic, Optional, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.envs import IdentityEnv

T = TypeVar("T", int, np.ndarray)

class IdentityEnv(gym.Env, Generic[T]):
    def __init__(self, dim: Optional[int] = None, space: Optional[spaces.Space] = None, ep_length: int = 100):
        """
        Identity environment for testing purposes

        :param dim: the size of the action and observation dimension you want
            to learn. Provide at most one of ``dim`` and ``space``. If both are
            None, then initialization proceeds with ``dim=1`` and ``space=None``.
        :param space: the action and observation space. Provide at most one of
            ``dim`` and ``space``.
        :param ep_length: the length of each episode in timesteps
        """
        if space is None:
            if dim is None:
                dim = 1
            space = spaces.Discrete(dim)
        else:
            assert dim is None, "arguments for both 'dim' and 'space' provided: at most one allowed"

        self.action_space = self.observation_space = space
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[T, dict]:
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        self.num_resets += 1
        self._choose_next_state()
        return self.state, {}

    def step(self, action: T) -> tuple[T, float, bool, bool, dict[str, Any]]:
        reward = self._get_reward(action)
        self._choose_next_state()
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.state, reward, terminated, truncated, {}

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

    def _get_reward(self, action: T) -> float:
        return 1.0 if np.all(self.state == action) else 0.0

    def render(self, mode: str = "human") -> None:
        pass


class InvalidActionEnvDiscrete(IdentityEnv[int]):
    """
    Identity env with a discrete action space. Supports action masking.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dim is None:
            dim = 1
        assert n_invalid_actions < dim, f"Too many invalid actions: {n_invalid_actions} < {dim}"

        space = spaces.Discrete(dim)
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(space.n, dtype=int)
        self.invalid_actions: list[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()
        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i != self.state]
        self.invalid_actions = np.random.choice(  # type: ignore[assignment]
            potential_invalid_actions, self.n_invalid_actions, replace=False
        ).tolist()

    def action_masks(self) -> list[bool]:
        mask = [action not in self.invalid_actions for action in self.possible_actions]
        return mask


class InvalidActionEnvMultiDiscrete(IdentityEnv[np.ndarray]):
    """
    Identity env with a multidiscrete action space. Supports action masking.
    """

    action_space: spaces.MultiDiscrete

    def __init__(
        self,
        dims: Optional[list[int]] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dims is None:
            dims = [1, 1]

        if n_invalid_actions > sum(dims) - len(dims):
            raise ValueError(f"Cannot find a valid action for each dim. Set n_invalid_actions <= {sum(dims) - len(dims)}")

        space = spaces.MultiDiscrete(dims)
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(sum(dims))
        self.invalid_actions: list[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

        converted_state: list[int] = []
        running_total = 0
        for i in range(len(self.action_space.nvec)):
            converted_state.append(running_total + self.state[i])
            running_total += self.action_space.nvec[i]

        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i not in converted_state]
        self.invalid_actions = np.random.choice(  # type: ignore[assignment]
            potential_invalid_actions, self.n_invalid_actions, replace=False
        ).tolist()

    def action_masks(self) -> list[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]


class InvalidActionEnvMultiBinary(IdentityEnv[np.ndarray]):
    """
    Identity env with a multibinary action space. Supports action masking.
    """

    def __init__(
        self,
        dims: Optional[int] = None,
        ep_length: int = 100,
        n_invalid_actions: int = 0,
    ):
        if dims is None:
            dims = 1

        if n_invalid_actions > dims:
            raise ValueError(f"Cannot find a valid action for each dim. Set n_invalid_actions <= {dims}")

        space = spaces.MultiBinary(dims)
        self.n_dims = dims
        self.n_invalid_actions = n_invalid_actions
        self.possible_actions = np.arange(2 * dims)
        self.invalid_actions: list[int] = []
        super().__init__(space=space, ep_length=ep_length)

    def _choose_next_state(self) -> None:
        self.state = self.action_space.sample()

        converted_state: list[int] = []
        running_total = 0
        for i in range(self.n_dims):
            converted_state.append(running_total + self.state[i])
            running_total += 2

        # Randomly choose invalid actions that are not the current state
        potential_invalid_actions = [i for i in self.possible_actions if i not in converted_state]
        self.invalid_actions = np.random.choice(  # type: ignore[assignment]
            potential_invalid_actions, self.n_invalid_actions, replace=False
        ).tolist()

    def action_masks(self) -> list[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]
