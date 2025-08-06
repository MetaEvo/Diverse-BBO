from typing import Any, Callable, List, Optional

import gym, warnings
import numpy as np

from VectorEnv.worker import EnvWorker


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env], no_warning=False) -> None:
        self.env = env_fn()
        if no_warning:
            warnings.filterwarnings("ignore")
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return self.env.get_env_attr(key)

    def set_env_attr(self, key: str, value: Any) -> None:
        self.env.set_env_attr(key, value)
        
    def get_env_obj(self):
        return self.env

    def reset(self) -> Any:
        return self.env.reset()

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray]) -> None:
        self.result = self.env.step(action)
        
    def customized_method(self, func: str, data) -> Any:
        self.result = eval('self.env.'+func)(**data)
            
    def send_reset(self) -> None:
        self.result = self.env.reset()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        # super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        if hasattr(self.env, 'close'):
            self.env.close()
