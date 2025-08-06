"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`gplearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import deepcopy
from datetime import datetime
import gc
import itertools
from abc import ABCMeta, abstractmethod
import re
from time import time
from warnings import warn

from tqdm import tqdm

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.utils.multiclass import check_classification_targets


from ._program import _Program, print_formula
from .fitness import _fitness_map, _Fitness
from .functions import _function_map, _Function, sig1 as sigmoid
from .utils import _partition_estimators
from .utils import check_random_state
import os
import ray
__all__ = ['SymbolicRegressor']

MAX_INT = np.iinfo(np.int32).max  # int32类型的最大值



# 用于去除 ANSI 转义序列的正则表达式
def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

@ray.remote(num_cpus = 1,num_gpus = 0 )
def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params,n_jobs,gen,fid):
    print(f"func {fid} : in gen {gen} njobs: {n_jobs}  started !!")
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']  # 锦标赛规模，目前都是20，应该要根据种群大小来调整
    function_set = params['function_set']  # 数组，所有不同的函数的_Function对象数组
    arities = params['arities']  # 字典，arity-_Function对象数组 键值对
    init_depth = params['init_depth']  # 初始种群深度限制
    mutate_depth = params['mutate_depth']  # 突变产生的树的深度限制
    init_method = params['init_method']  # 树的生成方法，full，grow或者half
    # const_range = params['const_range']
    variable_range = params['variable_range']  # 变量取值范围，用于限制常数主导现象
    metric = params['_metric']  # _Fitness对象列表，适应度计算函数
    transformer = params['_transformer']
    parsimony_coefficient = params['parsimony_coefficient']  # 简约系数
    method_probs = params['method_probs']  # 突变的概率分布列
    p_point_replace = params['p_point_replace']  # 点突变中每个点发生突变的概率，目前是0.05
    max_samples = params['max_samples']  # 整个样本集不采样占比的最大比例
    feature_names = params['feature_names']  # 输入向量的每个分量的名称

    problemID = params['problemID']
    problem_coord = params['problem_coord']
    model = params['model']
    scaler = params['scaler']
    save_path = params['save_path']
    
    # 此处代码是搜索一个problemID的函数 多线程并行搜索
    # 根据problemID读取problem对应的 2维 坐标


    max_samples = int(max_samples * n_samples)  # 不采样的数量

    def _tournament():  # 锦标赛筛选
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)  # 从现有种群中挑选足够数量的竞争者来进行锦标赛筛选
        fitness = [parents[p].raw_fitness_ for p in contenders]  # 计算竞争者的适应度
        # fitness = [parents[p].fitness_ for p in contenders]  # 计算竞争者的适应度
        if metric.greater_is_better:  # 确定metric是最小化优化还是最大化优化
            parent_index = contenders[np.argmax(fitness)]  # 选出适应度最大的
        else:
            parent_index = contenders[np.argmin(fitness)]  # 选出适应度最小的
        return parents[parent_index], parent_index

    # Build programs
    programs = []
    with tqdm(range(n_programs),desc=f'Func{fid} job{n_jobs} in gen {gen}') as pbar:
        for i in range(n_programs):  # n_programs是种群大小
            random_state = check_random_state(seeds[i])

            if parents is None:  # 第一代的parents都是None，也就不会进行交叉或突变
                program = None  # program为None，包装成_Program对象时会调用build_program函数来生成program
                genome = None  # 产生方法信息
            else:
                method = random_state.uniform()  # 随机选择一种基因变化方式
                if  method < method_probs[0] :
                    action = 'Crossover'
                elif method < method_probs[1]:
                    action = 'Subtree Mutation'
                elif method < method_probs[2]:
                    action = 'Hoist Mutation'
                elif method < method_probs[3]:
                    action = 'Point Mutation'
                else :
                    action = 'Reproduction'
                    
                    
                # print(f"func {fid} : in gen{gen} njobs: {n_jobs}\t choose for individual {i + 1} / {n_programs} : {action} started !!")
                parent, parent_index = _tournament()

                if method < method_probs[0]:  # crossover
                    donor, donor_index = _tournament()  # 通过锦标赛选择较优个体来作为donor
                    program, removed, remains = parent.crossover(donor.program,
                                                                random_state)
                    genome = {'method': 'Crossover',
                            'parent_idx': parent_index,
                            'parent_nodes': removed,
                            'donor_idx': donor_index,
                            'donor_nodes': remains}
                elif method < method_probs[1]:  # subtree
                    program, removed, _ = parent.subtree_mutation(random_state)  # 新的随机产生的个体作为donor
                    genome = {'method': 'Subtree Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': removed}
                elif method < method_probs[2]:  # hoist
                    program, removed = parent.hoist_mutation(random_state)
                    genome = {'method': 'Hoist Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': removed}
                elif method < method_probs[3]:  # point
                    program, mutated = parent.point_mutation(random_state)
                    genome = {'method': 'Point Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': mutated}
                else:  # 不变
                    # reproduction
                    program = parent.reproduce()
                    genome = {'method': 'Reproduction',
                            'parent_idx': parent_index,
                            'parent_nodes': []}
    
                # print(f"func {fid} : in gen{gen} njobs: {n_jobs}\t choose for individual {i + 1} / {n_programs} : {genome['method']}  finished !!")
            program = _Program(function_set=function_set,
                            arities=arities,
                            init_depth=init_depth,
                            mutate_depth=mutate_depth,
                            init_method=init_method,
                            n_features=n_features,
                            metric=metric,
                            transformer=transformer,
                            variable_range=variable_range,
                            p_point_replace=p_point_replace,
                            parsimony_coefficient=parsimony_coefficient,
                            feature_names=feature_names,
                            random_state=random_state,
                            program=program,# 将program包装为_Program对象
                            problemID=problemID,# 采样点的id
                            problem_coord = problem_coord,#采样点的2D坐标
                            model = model,
                            scaler = scaler,
                            save_path = save_path)
            program.parents = genome

            # Draw samples, using sample weights, and then fit
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))  # sample_weight全为1
            else:
                curr_sample_weight = sample_weight.copy()
            oob_sample_weight = curr_sample_weight.copy()  # out of bag fitness计算时使用的权重

            indices, not_indices = program.get_all_indices(n_samples,
                                                        max_samples,
                                                        random_state)

            curr_sample_weight[not_indices] = 0  # 不采样的样本权重设为0
            oob_sample_weight[indices] = 0  # 是curr的反相

            start_time = time()
            program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight,random_state=random_state)  # 计算加权平均值时的权重
            # if max_samples < n_samples:  # 若存在降采样，那么也计算out of bag fitness
            #     # Calculate OOB fitness
            #     program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight,random_state=random_state)
            # print(f"func {fid} : in gen {gen} njobs: {n_jobs}\tfor individual {i + 1} / {n_programs} finished ela calculate in {time() - start_time}s !!")
            programs.append(program)  # 添加到新一代的种群中
            pbar.set_postfix({
                        'ind_fitness' : program.raw_fitness_,
                        'ind_depth': program.depth_,
                        'ela_best_dim': program.best_dim,
                        'ela_cal_time': time() - start_time,
                        # 'oob_fitness' :self.run_details_['best_oob_fitness'][-1]
                })    
            pbar.update(1)

    return programs  # 返回新一代种群


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):  # !!!

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 variable_range=(-1., 1.),
                 init_depth=(2, 6),
                 mutate_depth=None,
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 problemID = 0,
                 problem_coord = None,
                 model = None,
                 dim = 10,
                 scaler = None,
                 save_path = None,
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.variable_range = variable_range
        self.init_depth = init_depth
        self.mutate_depth = mutate_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.problemID = problemID
        self.problem_coord = problem_coord
        self.dim = dim
        self.save_path = save_path
        # 此处对于每个regressor，都需要一个AE model 用于计算population内所有个体的ela降维坐标
        self.model = model
        self.scaler = scaler

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        X, y = self._validate_data(X, y, y_numeric=True)

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []  # _Function对象数组
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}  # 点突变中arity相同的函数之间相互突变
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        # 将metric函数名映射为_Fitness对象
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        # 突变概率分布列
        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)  # 累积概率分布函数

        # 参数合法性检验
        if self._method_probs[-1] > 1:  # 若概率总和超过1，报错
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not((isinstance(self.variable_range, tuple) and
                len(self.variable_range) == 2) or self.variable_range is None):
            raise ValueError('variable_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.mutate_depth is not None and (not isinstance(self.mutate_depth, tuple) or
                                              len(self.mutate_depth) != 2):
            raise ValueError('mutate_depth should be a tuple with length two.')
        if self.mutate_depth is not None and self.mutate_depth[0] > self.mutate_depth[1]:
            raise ValueError('mutate_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_in_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_in_,
                                    len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either '
                                 '"sigmoid" or _Function object, got %s' %
                                 type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, '
                                 'got %d.' % (self._transformer.arity))

        params = self.get_params()  # 将所有属性值打包成字典
        # print(params)
        # _开头的属性是内部数据结构，没有可读性
        params['_metric'] = self._metric
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        # 需要添加进params 的 problemID,problem的坐标 和 model
        params['problemID'] = self.problemID
        params['problem_coord'] = self.problem_coord
        params['model'] = self.model
        params['scaler'] = self.scaler
        params['save_path'] = self.save_path
        
        if not self.warm_start or not hasattr(self, '_programs'):# 若没有初始种群输入
            # Free allocated memory, if any
            self._programs = []  # 记录每一代种群的所有个体，每个种群是其中一个元素
            self.run_details_ = {'generation': [],
                                 'average_depth': [],
                                 'average_fitness': [],
                                 'best_depth': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)  # 已迭代代数
        n_more_generations = self.generations - prior_generations  # 剩余迭代次数

        # 对热启动的种群输入的合法性进行检验
        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        reinit_frequency = 5  # 每5代检测一次是否需要重初始化
        reinit_counter = 5
        reinit = False
        best_10_program = []
        best_10_txt = []
        # 用于重初始化记录fitness
        best_fitness_list = []
        with tqdm(range(prior_generations,self.generations),desc=f'Searching GP_Funtion_{self.problemID} ') as pbar:
            for gen in range(prior_generations, self.generations):  # 完成剩余迭代
                start_time = time()
                reinit_counter -= 1
                if gen == 0 or reinit:
                    parents = None
                else:
                    parents = self._programs[gen - 1]
                if not reinit:
                    # Parallel loop, 开始训练
                    # n_jobs为实际并行所使用的cpu数，n_programs是每个job负责的作业量列表，starts为每个job的开头在种群中的索引的数组
                    n_jobs, n_programs, starts = _partition_estimators(
                        self.population_size, self.n_jobs)
                    seeds = random_state.randint(MAX_INT, size=self.population_size)
                    #2024.11.2
                    # 改成ray来实现并行
                    population_biglist = [_parallel_evolve.remote(n_programs[i],parents,X,y,sample_weight,seeds[starts[i]:starts[i + 1]],params,i,gen,self.problemID) for i in range(n_jobs) ]
                    population = ray.get(population_biglist)

                    # Reduce, maintaining order across different n_jobs
                    population = list(itertools.chain.from_iterable(population))  # 将可迭代对象组成的数组整合为一个数组
                else:
                    print('Reinitialization.')
                    reinit = False
                    population = []
                    
                    n_jobs, n_programs, starts = _partition_estimators(
                        self.population_size , self.n_jobs)  
                    seeds = random_state.randint(MAX_INT, size=self.population_size )
                    
                    population_biglist = [_parallel_evolve.remote(n_programs[i],parents,X,y,sample_weight,seeds[starts[i]:starts[i + 1]],params,i,gen,self.problemID) for i in range(n_jobs) ]
                    new_population = ray.get(population_biglist)
                    # Reduce, maintaining order across different n_jobs
                    # 将可迭代对象组成的数组整合为一个数组
                    new_population = list(itertools.chain.from_iterable(new_population))
                    population = population + new_population


                fitness = [program.raw_fitness_ for program in population]
                length = [program.length_ for program in population]
                depth = [program.depth_ for program in population]
                parsimony_coefficient = None
                if self.parsimony_coefficient == 'auto':  # 简约系数自动计算公式
                    parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                            np.var(length))
                for program in population:
                    program.fitness_ = program.fitness(parsimony_coefficient)


                
                self._programs.append(population)
                
                # Remove old programs that didn't make it into the new population.
                # _programs记录了以往种群的信息，但这里有存储优化，要做重初始化可能需要改这里
                if not self.low_memory:  # 默认是False
                    if gen - reinit_frequency >= 0:
                        self._programs[gen - reinit_frequency] = None
                    for old_gen in np.arange(gen, max(0, gen - reinit_frequency + 1), -1):  # 创建等差数组，公差是-1
                        indices = []
                        for program in self._programs[old_gen]:
                            # 在上个循环中被设为None的program的父母program不添加进indices中
                            if program is not None and program.parents is not None:
                                for idx in program.parents:
                                    if 'idx' in idx:
                                        indices.append(program.parents[idx])   # 记录父母program的索引值
                        indices = set(indices)  # 转为集合，排除掉重复的元素
                        for idx in range(self.population_size):
                            if idx not in indices:
                                self._programs[old_gen - 1][idx] = None  # 去掉上一代对新一代种群没有影响的个体
                # elif gen > 0:  # low_memory为真，则不保留以往种群

                
                if reinit_counter <= 0:
                    reinit_counter = reinit_frequency
                    # 2024.11.2直接用一个list存前n代的信息 , 无需多次拿前代个体
                    # 重初始化必要性检测，评判标准是前(reinit_frequency - 2)代种群适应度平均值变化很小就重新初始化
                    best_fitness_list = []
                    start, end = gen, gen - reinit_frequency
                    for old_gen in np.arange(start, end, -1):  # 创建等差数组，公差是-1
                        fitness_list = []
                        for program in self._programs[old_gen]:
                            if program is not None:
                                fitness_list.append(program.raw_fitness_)
                        if old_gen == start:
                            last_fitness = fitness_list
                        if self._metric.greater_is_better:
                            best_fitness_list.append(np.max(fitness_list))
                        else:
                            best_fitness_list.append(np.min(fitness_list))
                    # print(f'best_fitness_list: {best_fitness_list}')
                    # print(f'standard variance:{np.std(best_fitness_list)}')
                    # print(f'average:{np.average(best_fitness_list)}')
                    # print(f'metric:{np.std(best_fitness_list) / np.average(best_fitness_list)}')
                    # 标准差远小于平均值，则只保留最后一代前几个最优个体
                    if np.std(best_fitness_list) / np.average(best_fitness_list) < 0.1:
                        reinit = True
                    #     last_fitness = [program.raw_fitness_ for program in self._programs[gen] if program is not None]
                    # best_fitness_list = []
                    
                    
                # Record run details
                if self._metric.greater_is_better:  # 求最优个体
                    best_program = population[np.argmax(fitness)]
                else:
                    best_program = population[np.argmin(fitness)]
                
                
                self.run_details_['generation'].append(gen)
                self.run_details_['average_depth'].append(np.mean(depth))
                self.run_details_['average_fitness'].append(np.mean(fitness))
                self.run_details_['best_depth'].append(best_program.depth_)
                self.run_details_['best_fitness'].append(best_program.raw_fitness_)
                oob_fitness = np.nan
                if self.max_samples < 1.0:
                    oob_fitness = best_program.oob_fitness_  # 在_parallel_evolve中计算
                self.run_details_['best_oob_fitness'].append(oob_fitness)
                generation_time = time() - start_time
                self.run_details_['generation_time'].append(generation_time)

                pbar.set_postfix({
                        'avg_depth' : self.run_details_['average_depth'][-1],
                        'avg_fitness': self.run_details_['average_fitness'][-1],
                        'best_depth': self.run_details_['best_depth'][-1],
                        'best_fitness': self.run_details_['best_fitness'][-1],
                        # 'oob_fitness' :self.run_details_['best_oob_fitness'][-1]
                })    
                pbar.update(1)
                # save functions in every genaration
                sorted_indices = np.argsort(fitness)[:20]
                # save best 10 function in every gen
                for i in range(1): 
                    best_10_program.append(deepcopy(population[sorted_indices[i]]))
                    # 顺带记录对应的代数gen
                    best_10_txt.append((deepcopy(fitness[sorted_indices[i]]),deepcopy(population[sorted_indices[i]].depth_),\
                        gen,deepcopy(population[sorted_indices[i]].coordi_2D),deepcopy(population[sorted_indices[i]].best_dim)))      
                # Check for early stopping
                if self._metric.greater_is_better:
                    best_fitness = fitness[np.argmax(fitness)]
                    if best_fitness >= self.stopping_criteria:
                        break
                else:
                    best_fitness = fitness[np.argmin(fitness)]
                    if best_fitness <= self.stopping_criteria:
                        break

        # 迭代停止后找最后一代的最优个体
            # Find the best individual in the final generation
        if self._metric.greater_is_better:
            self._program = self._programs[-1][np.argmax(fitness)]
        else:
            # sorted_indices = np.argsort(fitness)[:20]
            indexed_lst = list(enumerate(best_10_txt))
            sorted_lst_with_indices = sorted(indexed_lst, key=lambda x: x[1][0])
            sorted_lst = [item for index, item in sorted_lst_with_indices]
            sorted_idx =[index for index, item in sorted_lst_with_indices]
            
            # 保存最后每代最好的10个个体
            best_10_program = [best_10_program[i] for i in sorted_idx]
            # 保存对应的txt信息
            best_str_list = []
            for idx,pair in enumerate(sorted_lst):
                best_str_list.append(f'fitness : {pair[0]} in {pair[4]}D'+'\t'+f'depth {pair[1]}\t' +f'Gen {pair[2]}\t 2D coordi : {pair[3]}\t' + remove_ansi_escape_sequences(print_formula(best_10_program[idx].program, self.dim,show_operand = True,no_print = True)) + '\n')
            
            self._program = self._programs[-1][np.argmin(fitness)]

        return  {f'func{self.problemID}_{X.shape[-1]}D_{self.population_size}size_{self.generations}gens.pickle': best_10_program,\
                f'func{self.problemID}_{X.shape[-1]}D_{self.population_size}size_{self.generations}gens.txt': best_str_list}


class SymbolicRegressor(BaseSymbolic, RegressorMixin):  # !!!

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 *,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 variable_range=(-1., 1.),
                 init_depth=(2, 6),
                 mutate_depth=None,
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 problemID = 0,
                 problem_coord = None,
                 model = None,
                 dim = 10,
                 scaler = None,
                 save_path = None,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            variable_range=variable_range,
            init_depth=init_depth,
            mutate_depth=mutate_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            problemID = problemID,
            problem_coord = problem_coord,
            model = model,
            dim = dim,
            scaler = scaler,
            save_path=save_path,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')
        # 检查输入X是不是numpy数组或稀疏矩阵，是稀疏矩阵则转换为Compressed Sparse Row格式，若不是稀疏矩阵则转为numpy数组
        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        y = self._program.execute(X)

        return y


