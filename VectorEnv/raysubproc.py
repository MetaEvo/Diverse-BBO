from typing import Any, Callable, List, Optional, Tuple, Union

import gym, os, platform
import numpy as np

from VectorEnv.worker import EnvWorker
import ray, copy, warnings
from dill import dumps, loads


class _SetAttrWrapper(gym.Wrapper):

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env, key, value)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)


class RaySubprocEnvWorker(EnvWorker):
    """Ray worker used in RayVectorEnv."""
    num_cpu_per_worker = 1
    num_gpu_per_worker = 0

    def __init__(self, env_fns: Callable[[], gym.Env], worker_fn, num_cpu_per_worker=1, num_gpu_per_worker=0, no_warning=False) -> None:
        # self.env = ray.remote(_SetAttrWrapper).options(num_cpus=num_cpus).remote(env_fn())
        # self.env = env_fn()
        # self.workers = worker_fn(env_fns)
        self.env_num = len(env_fns)
        self.worker_fn = worker_fn
        self.envs = [fn() for fn in env_fns]
        self.num_cpu_per_worker = num_cpu_per_worker
        self.num_gpu_per_worker = num_gpu_per_worker
        self.no_warning = no_warning
        super().__init__(env_fns)

    def get_env_attr(self, key: str, id=None) -> Any:
        if id is None:
            id = list(range(self.env_num))
        results = []
        for i in id:
            results.append(getattr(self.envs[i], key) if hasattr(self.envs[i], key) else None)
        return results
    
    def set_env_attr(self, key: str, value: Any, id=None) -> None:
        # ray.get(self.env.set_env_attr.remote(key, value))
        if id is None:
            id = list(range(self.env_num))
        for i, j in enumerate(id):
            setattr(self.envs[j], key, value[i])

    def send_reset(self, id=None) -> Any:
        if id is None:
            id = list(range(self.env_num))
        self.result = self.ray_reset.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(self.worker_fn, loads(dumps(self.envs)), id, self.no_warning)
    
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_reset(worker_fn, envs, id, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        envs = loads(dumps(envs))
        env_fns = [lambda e=p: e for p in envs]
        workers = worker_fn(env_fns)
        results = workers.reset(id)
        envs = workers.get_env_obj()
        workers.close()
        return results, loads(dumps(envs))
    
    def customized_method(self, func: str, data, id=None) -> Any:
        if id is None:
            id = list(range(self.env_num))
        self.result = self.ray_customized.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(self.worker_fn, self.envs, func, data, id, self.no_warning)
        
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_customized(worker_fn, envs, func, data, id, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        env_fns = [lambda e=p: e for p in envs]
        workers = worker_fn(env_fns)
        results = workers.customized_method(func, data, id)
        envs = workers.get_env_obj()
        workers.close()
        return results, envs

    @staticmethod
    def wait(  # type: ignore
        workers: List["RaySubprocEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["RaySubprocEnvWorker"]:
        results = [x.result for x in workers]
        ready_results, _ = ray.wait(results, num_returns=wait_num, timeout=timeout)
        return [workers[results.index(result)] for result in ready_results]

    def send(self, action: Optional[np.ndarray], id: Optional[np.ndarray]=None) -> None:
        # self.action is actually a handle
        if id is None:
            id = list(range(self.env_num))
        self.result = self.ray_step.options(num_cpus=self.num_cpu_per_worker, num_gpus=self.num_gpu_per_worker).remote(self.worker_fn, loads(dumps(self.envs)), action, id, self.no_warning)
    
    @ray.remote(num_cpus=num_cpu_per_worker, num_gpus=num_gpu_per_worker)
    def ray_step(worker_fn, envs, action, id, no_warning):
        if no_warning:
            warnings.filterwarnings("ignore")
        envs = loads(dumps(envs))
        env_fns = [lambda e=p: e for p in envs]
        workers = worker_fn(env_fns)
        results = workers.step(action, id)
        envs = workers.get_env_obj()
        workers.close()
        return results, loads(dumps(envs))

    def recv(
        self
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        results, self.envs = ray.get(self.result)
        return results

    def seed(self, seed: Optional[int] = None) -> List[int]:
        super().seed(seed)
        # return ray.get(self.env.seed.remote(seed))
        for i in range(self.env_num):
            self.envs[i].seed(seed)

    def render(self, **kwargs: Any) -> Any:
        return [self.envs[i].render(**kwargs) for i in range(self.env_num)]

    def close_env(self) -> None:
        pass
