from typing import Any, Callable, List, Optional, Tuple, Union

import gym, copy, warnings, torch
import numpy as np

from VectorEnv.worker import EnvWorker
from dill import dumps, loads
try:
    import ray
except ImportError:
    pass


class _SetAttrWrapper(gym.Wrapper):

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env, key, value)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)


class RayEnvWorker(EnvWorker):
    """Ray worker used in RayVectorEnv."""
    num_cpu_per_worker = 1
    num_gpu_per_worker = 0

    def __init__(self, env_fn: List[Callable[[], gym.Env]], num_cpu_per_worker: int=1, num_gpu_per_worker: int=0, no_warning=False) -> None:
        # self.env = ray.remote(_SetAttrWrapper).options(num_cpus=num_cpus).remote(env_fn())
        self.env = env_fn()
        self.num_cpu_per_worker = num_cpu_per_worker
        self.num_gpu_per_worker = num_gpu_per_worker
        self.no_warning = no_warning
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return self.env.get_env_attr(key)
    
    # @ray.remote(num_cpus=1, num_gpus=0)
    # def ray_getattr(env, key):
    #     return getattr(env, key) if hasattr(env, key) else None

    def set_env_attr(self, key: str, value: Any) -> None:
        # ray.get(self.env.set_env_attr.remote(key, value))
        self.env.set_env_attr(key, value)
        
    def get_env_obj(self):
        return self.env

    def send_reset(self) -> Any:
        self.result = self.ray_reset.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(loads(dumps(self.env)), self.no_warning)
    
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_reset(env, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        env = loads(dumps(env))
        state = env.reset()
        return state, loads(dumps(env))

    @staticmethod
    def wait(  # type: ignore
        workers: List["RayEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["RayEnvWorker"]:
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results, num_returns=wait_num, timeout=timeout)
        return [workers[results.index(result)] for result in ready_results]

    def send(self, action: Optional[np.ndarray]) -> None:
        # self.action is actually a handle
        self.result = self.ray_step.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(loads(dumps(self.env)), action, self.no_warning)
    
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_step(env, action, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        env = loads(dumps(env))
        results = env.step(action)
        return *results, loads(dumps(env))
    
    def customized_method(self, func: str, data) -> Any:
        self.result = self.ray_customized.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(loads(dumps(self.env)), func, data, self.no_warning)
    
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_customized(env, func, data, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        env = loads(dumps(env))
        results = eval('env.'+func)(**data)
        return results, loads(dumps(env))
        
    def recv(
        self
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        results = list(ray.get(self.result))
        self.env = results[-1]
        return results[:-1]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        super().seed(seed)
        # return ray.get(self.env.seed.remote(seed))
        self.env.seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        if hasattr(self.env, 'close'):
            self.env.close()
