"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import time
from datetime import timedelta
import math
import random
from copy import copy, deepcopy

from dataset.GP import GP_problem
from net.AE import get_encoded
import numpy as np
import pandas as pd
import os
from sklearn.utils.random import sample_without_replacement

from pflacco_v1.sampling import create_initial_sample

from .functions import _Function, _sum, _prod, _mean, _sigmoid, _protected_power,_tanh, sum, prod, mean, min2, max2
from .functions import _protected_division, _protected_sqrt, _protected_log, _protected_inverse, _protected_exp , add2 , mul2 , sub2 
from .utils import check_random_state
from .ela_feature import get_ela_feature



check_constant_function = False

default_remaining = [4, 1, 1, 1]
default_total = [0, 0, 0, 0]
aggregate = ['prod', 'mean', 'sum']  # aggragate函数名称列表
concatenate = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean', 'min', 'max']  # 可能会出现主导问题的连接操作符
operator = ['neg', 'inv', 'abs']  # neg/inv/abs
elementary_functions = ['sin', 'cos', 'tan', 'log','tanh']  # 基本初等函数名称列表
ignore = [['add', 'sub', 'sum'],  # 加性函数节点
          ['add', 'sub', 'sum', 'mul', 'div', 'prod']]  # 加性和乘性函数节点

def validate_interval(interval):
    return interval[0] < interval[1]

def printout(program, max_dim):
    for node in program:
        if isinstance(node, _Function):
            if node.name in ['sum', 'prod', 'mean']:
                print(f"{node.name}[{node.input_dimension},{node.arity}]",
                      end=' ')  # ,{node.parent_distance},{node.child_distance_list}
            else:
                print(f"{node.name}[{node.input_dimension}]",
                      end=' ')  # [{node.parent_distance},{node.child_distance_list}]
        elif isinstance(node, tuple):  # 变量节点
            print(f'({node[0], max_dim - node[1], node[2]})', end=' ')
        else:
            print(node, end=' ')
    print()


def print_formula(program, max_dim, show_operand=False, no_print=False):  # 颜色编码从'\033[31m'到'\033[38m'
    formula_stack = []
    min_priority_stack = []  # 用于子树的内括号判断，记录每个子树的min_priority
    # name_mapping = {'add': '+', 'sub': '-', 'mul': '×', 'div': '/', 'pow': '^'}  # , 'sum': '+', 'prod': '×'
    name_mapping = {'add': '+', 'neg': '-', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '^'}  # , 'sum': '+', 'prod': '×'
    # priority = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
    priority = {'add': 1, 'neg': 2, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
    formula = ''
    last_arity = 0
    last_name = ''
    min_priority = 5  # 用于子树的外括号判断
    for node in program:
        if isinstance(node, _Function):
            formula_stack.append(node.name)
            formula_stack.append(node.arity)
            last_name = node.name
            last_arity = node.arity
        else:
            if show_operand:  # 展示具体的操作数，操作数分为向量切片(tuple)和常数向量list[ndarray]两种
                temp = '\033[36m' + '[' + '\033[0m'
                if isinstance(node, tuple):
                    for i in range(node[0], max_dim - node[1], node[2]):
                        temp += 'X' + str(i) + ', '
                else:
                    for i in node[0]:  # 遍历ndarray
                        temp += str(i) + ', '
                temp = temp[:-2] + '\033[36m' + ']' + '\033[0m'
                formula_stack.append(temp)
            else:
                formula_stack.append('o')
            # 如果arity已经满足，且中间没有arity数字，说明操作数数目已经满足
            # 同时在该完整子树内进行内括号判断，即根据基本操作符的优先级顺序来添加括号
            while last_arity + 1 <= len(formula_stack) and \
                    formula_stack[-(last_arity + 1)] == last_arity and\
                    formula_stack[-(last_arity + 1)] not in formula_stack[- last_arity:]:
                for i in range(last_arity):  # 移除末尾last_arity个操作数
                    intermediate = formula_stack.pop()
                    if intermediate[0] != '@':  # 不是子树，则不需要内括号判断
                        formula = intermediate + formula
                        if last_name == 'neg':
                            min_priority = 0  # neg对外优先级最低，对内优先级与sub相同
                        elif last_name in name_mapping.keys():
                            min_priority = priority[last_name]
                        else:
                            min_priority = 5
                    else:  # 如果是子树，则需要知道min_priority
                        intermediate = intermediate[1:]  # 去掉第一个特殊字符@
                        min_priority = min_priority_stack.pop()  # 获取最后一个子树的min_priority
                        if last_name in name_mapping.keys():  # 在函数名字-符号映射表内，则判断是否需要添加括号
                            if min_priority < priority[last_name] or \
                                    min_priority == priority[last_name] and min_priority % 2 == 0:  # 相等且为2或4，即减和除
                                formula = '(' + intermediate + ')' + formula
                            else:
                                formula = intermediate + formula
                            if last_name != 'neg':
                                min_priority = priority[last_name]  # 优先级取最低的
                            else:
                                min_priority = 0
                        else:  # 其他函数，无需内括号判断
                            formula = intermediate + formula
                            min_priority = 5
                    if i != last_arity - 1:  # 不是最后一个操作数，则需要加上操作符
                        if last_name in name_mapping.keys():
                            formula = name_mapping[last_name] + formula
                        else:
                            formula = ', ' + formula
                    elif last_name == 'neg':
                        formula = name_mapping[last_name] + formula
                formula_stack.pop()  # 移除函数节点的arity数字
                formula_stack.pop()  # 移除函数节点的函数名字
                # 一个完整子树的外括号判断：abs为||；neg为-，并且判断是否需要添加括号；非基本操作符的函数外括号添加，聚集函数使用{}，其余使用()
                if last_name == 'abs':
                    front = '\033[32m' + '|' + '\033[0m'
                    end = '\033[32m' + '|' + '\033[0m'
                    formula = front + formula + end
                # elif last_name == 'neg':
                #     if min_priority <= 2:
                #         formula = '-(' + formula + ')'
                #     else:
                #         formula = '-' + formula
                #     min_priority = 0
                elif last_name not in name_mapping.keys():  # 若不在函数名字-符号映射表内，则需要加上函数节点的名字
                    if last_name in aggregate:
                        front = '\033[31m' + '{' + '\033[0m'
                        end = '\033[31m' + '}' + '\033[0m'
                        formula = last_name + front + formula + end
                    else:
                        front = '('
                        end = ')'
                        formula = last_name + front + formula + end
                if len(formula_stack) == 0:  # formula为空
                    if not no_print:
                        print(formula)
                    return formula
                # 找新的最后一个函数节点，更新last_name和last_arity
                for index in range(len(formula_stack)):
                    if not isinstance(formula_stack[- 1 - index], str):  # 不是str，则为arity
                        last_arity = formula_stack[- 1 - index]
                        last_name = formula_stack[- 2 - index]
                        break
                formula = '@' + formula  # @开头表示这是一个子树的文本表示
                formula_stack.append(formula)  # 附加到末尾
                min_priority_stack.append(min_priority)
                formula = ''
                min_priority = 5


# function是_Function对象，若函数为连加或连乘，则需要创建新的实例，以记录不同的arity值。
def new_operator(function, random_state, n_features, output_dimension, remaining):
    new_function = 0
    new_remaining = deepcopy(remaining)
    # total = [0, 0, 0, 0]  # 默认值是[0, 0, 0, 0]
    if function.name == 'sum' or function.name == 'prod' or function.name == 'mean':
        if output_dimension == 1:  # output_dimension=1和arity=1是充分必要条件！
            arity = 1  # arity为1时可求向量的各分量的累加或累乘
        else:  # output_dimension不为1时arity也不为1
            if n_features > 1 :
                if function.name == 'sum':
                    arity = 2
                else :
                    # mean的话容易出现mean(x+x)，避免
                    arity = random_state.randint(2, 5)  # [2, self.n_features]
            else:
                arity = 2
        if function.name == 'sum':  # 初始名字是sum
            # new_function = _Function(function=_sum, name=f'sum({arity})', arity=arity)
            new_function = _Function(function=_sum, name=f'sum', arity=arity)
            # TODO 修改sum的权重为2
            new_remaining[0] -= 2  # 剩余aggregate次数 - 2
            # total[0] += 1
        elif function.name == 'prod':
            # new_function = _Function(function=_prod, name=f'prod({arity})', arity=arity)
            new_function = _Function(function=_prod, name=f'prod', arity=arity)
            new_remaining[0] -= 2  # 剩余aggregate次数 - 2
            # total[0] += 3
        elif function.name == 'mean':
            # new_function = _Function(function=_mean, name=f'mean({arity})', arity=arity)
            new_function = _Function(function=_mean, name=f'mean', arity=arity)
            # TODO 修改mean的权重为2
            new_remaining[0] -= 2  # 剩余aggregate次数 - 2
            # total[0] += 3
        # 输入/输出维度设置
        new_function.remaining = new_remaining
        # new_function.total = total
        new_function.output_dimension = output_dimension
        if new_function.arity == 1:  # arity为=1时可以允许输入维度不等于输出维度
            if n_features > 1:
                # new_function.input_dimension = random_state.randint(1, n_features) + 1  # [2, self.n_features]，至少为2
                # [self.n_features - 1, self.n_features]，切片的维度
                new_function.input_dimension = random_state.randint(n_features - 2, n_features) + 1
            else:
                new_function.input_dimension = 1
        else:
            new_function.input_dimension = new_function.output_dimension
    else:  # 对于其他运算符同样要创建新实例，以记录不同的输入和输出维度
        if function.name == 'add':
            new_function = _Function(function=np.add, name='add', arity=2)
        elif function.name == 'sub':
            new_function = _Function(function=np.subtract, name='sub', arity=2)
        elif function.name == 'mul':
            new_function = _Function(function=np.multiply, name='mul', arity=2)
        elif function.name == 'div':
            new_function = _Function(function=_protected_division, name='div', arity=2)
        elif function.name == 'max':
            new_function = _Function(function=np.maximum, name='max', arity=2)
            new_remaining[0] -= 1  # 剩余aggregate次数 - 1
            # total[0] += 1
        elif function.name == 'min':
            new_function = _Function(function=np.minimum, name='min', arity=2)
            new_remaining[0] -= 1  # 剩余aggregate次数 - 1
            # total[0] += 1
        elif function.name == 'pow':
            new_function = _Function(function=_protected_power, name='pow', arity=2)
            new_remaining[1] -= 1  # 剩余pow次数 - 1
            # total[1] += 1
        elif function.name == 'log':
            new_function = _Function(function=_protected_log, name='log', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'sin':
            new_function = _Function(function=np.sin, name='sin', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'cos':
            new_function = _Function(function=np.cos, name='cos', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'tan':
            new_function = _Function(function=np.tan, name='tan', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'tanh':
            new_function = _Function(function=_tanh, name='tanh', arity=1)
            new_remaining[2] -= 1  # 基本初等函数次数 - 1
            # total[2] += 1
        elif function.name == 'neg':
            new_function = _Function(function=np.negative, name='neg', arity=1)
            # new_remaining[3] = 1  # 还可以选inv和abs
        elif function.name == 'inv':
            new_function = _Function(function=_protected_inverse, name='inv', arity=1)
            # new_remaining[3] = 2  # 还可以选abs
        elif function.name == 'abs':
            new_function = _Function(function=np.abs, name='abs', arity=1)
            # new_remaining[3] = 3  # neg,inv和abs都不可再选
        elif function.name == 'sqrt':
            new_function = _Function(function=_protected_sqrt, name='sqrt', arity=1)
        elif function.name == 'sig':
            new_function = _Function(function=_sigmoid, name='sig', arity=1)
        elif function.name == 'exp':
            new_function = _Function(function=_protected_exp, name='exp', arity=1)
            new_remaining[3] -= 1  # 剩余exp次数减1
        # 这些运算符的输入维度=输出维度
        new_function.remaining = new_remaining
        # new_function.total = total
        new_function.output_dimension = output_dimension
        new_function.input_dimension = new_function.output_dimension
    return new_function


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    variable_range : tuple of two floats
        The range of variables to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,  # 函数(对象)集
                 arities,  # arity字典，对应上述函数集
                 init_depth,  # 第一代种群树深度范围
                 mutate_depth,  # 突变产生的树的深度范围限制
                 init_method,  # grow，full，half and half
                 n_features,  # 输入向量X的维度
                 variable_range,  # 变量的范围
                 metric,  # fitness
                 p_point_replace,  # 点突变的概率
                 parsimony_coefficient,  # 简约系数
                 random_state,  # np的随机数生成器
                 problemID, 
                 problem_coord, # 要搜索的program在二维空间中的坐标表示
                 model,# 用于降维的AE模型
                 scaler,# 用于对 gp ela进行归一化
                 save_path,# 存储生成的函数
                 transformer=None,  # sigmoid函数
                 feature_names=None,  # X的各分量名字
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.mutate_depth = mutate_depth
        self.init_method = init_method
        self.n_features = n_features
        self.variable_range = variable_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.problemID =problemID
        self.problem_coord = problem_coord
        self.model = model
        self.scaler = scaler
        self.save_path = save_path

        # 记录该问题最终使用哪个维度进行评估，fitness最靠近
        # 默认是开始的dim
        self.best_dim = n_features
        
        
        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state, output_dimension=1):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        output_dimension: int
            The dimension of program's output

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)  # 从给定的深度范围内随机选定该树的最大深度
        # init_function_set = [add2, sub2, mul2]  # , sum, prod  sum和prod作为根节点的arity只有1，不适合
        # 前两层节点改为从加减乘累加累乘中选择
        # function = random_state.randint(len(init_function_set))
        aggregate_function_set = []  # 聚合运算符  sum, prod, mean
        for item in self.function_set:
            if item.arity == 0:
                aggregate_function_set.append(item)
        if len(aggregate_function_set) == 0:
            raise ValueError('There should be at least one aggregate function in the function set.')
        init_function_set = [add2,mul2,sub2]  
        flag_root = random_state.randint(2)
        if flag_root:
            function = random_state.randint(len(init_function_set))
            function = init_function_set[function]
        else:
            function = random_state.randint(len(self.function_set))
            function = self.function_set[function]  # 随机挑选一个_Function对象
        current_remaining = default_remaining
        function = new_operator(function, random_state, self.n_features, output_dimension, current_remaining)
        function.depth = 0  # 初始节点的深度为0
        function.parent_distance = 0  # 根节点的该属性为0
        program = [function]
        terminal_stack = [function.arity]
        program = self.set_total(len(program) - 1, program)  # 设置total属性
        next_is_terminal = False
        next_is_function = False
        while terminal_stack:  # 当栈不为空时，重复添加函数或向量
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)  # 非函数节点和函数节点的概率受变量维度和函数类数影响
            choice = random_state.randint(choice)
            parent_index = self.find_parent(len(program), program)
            parent = program[parent_index]
            parent_name = parent.name  # 父节点函数名字
            existed_dimension = parent.input_dimension  # 上下层接口一致
            value_range = parent.value_range  # 父节点函数值域
            # 维护父节点的child_distance_list属性
            if not next_is_function and not next_is_terminal:  # 如果不是重复循环，就记录当前点
                parent.child_distance_list.append(len(program) - parent_index)
            # full策略优先选择添加函数
            # print(f'next_is_terminal:{next_is_terminal}, next_is_function:{next_is_function}')
            if (depth < max_depth) and (method == 'full' or choice <= len(self.function_set)) and not next_is_terminal \
                    or next_is_function:
                current_remaining = parent.remaining  # 记录父节点的remaining
                # current_function_set = self.clip_function_set(function_set=self.function_set,
                #                                               remaining=current_remaining,
                #                                               parent_name=parent_name)  # 求约束规则下的函数集
                if depth == 2 and existed_dimension == 1:  # 第两层若existed_dimension仍然是1，则选择aggregate运算符
                    current_function_set = self.clip_function_set(function_set=aggregate_function_set,
                                                                  remaining=current_remaining,
                                                                  parent_name=parent_name)  # 求约束规则下的函数集
                else:  # 前两层节点改为从加减乘累加累乘中选择
                    current_function_set = self.clip_function_set(function_set=self.function_set,
                                                                  remaining=current_remaining,
                                                                  parent_name=parent_name)  # 求约束规则下的函数集
                
                if 'exp' in current_function_set and existed_dimension == 1 :
                    current_function_set.remove('exp')
                # if parent_name in ['max', 'min'] and terminal_stack[-1] == 1:  # max和min的第二个操作数
                #     if isinstance(program[parent_index + 1], _Function):
                #         for item in current_function_set:  # current_function_set是_Function对象集合，但不利于根据函数名检索特定函数，改为函数名集合能优化检索时间
                #             if item.name == program[parent_index + 1].name:
                #                 current_function_set.remove(item)
                #         if program[parent_index + 1].name in current_function_set:
                #             current_function_set.remove(program[parent_index + 1].name)
                                
                
                temp_index = parent_index
                temp_parent = parent
                has_exp = False
                while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                    if temp_parent.name == 'exp' and temp_parent.input_dimension == 1:
                        has_exp = True
                        break
                    else:  # 不是sub和div
                        temp_index += temp_parent.parent_distance
                        temp_parent = program[temp_index]  # 父节点回溯
                if temp_parent.name == 'exp' and temp_parent.input_dimension == 1 and not has_exp:  # 若第一个sub或div节点是根节点
                    has_exp = True
                if has_exp:
                    # current_function_set是_Function对象集合，但不利于根据函数名检索特定函数，改为函数名集合能优化检索时间
                    for item in current_function_set:
                        if item.name == 'sum':  # exp下不出现sum节点
                            current_function_set.remove(item)
                            if len(current_function_set) == 0:  # 目前是aggregate函数集因为remaining和这个要求可能为空
                                for func in self.function_set:  # 改为函数名字集能优化时间复杂度
                                    if func.name == 'mean':  # 这要求func_set中一定要有mean，否则会出现next_is_function和next_is_terminal错误
                                        current_function_set.append(func)  # 但这样remaining属性的聚合函数剩余值会出现负数的情况，也许有影响
                                        break
                            break
                
                
                # 若函数集为空，或当前点pow函数的第二个操作数，则选择terminal
                if (parent_name == 'pow' and terminal_stack[-1] == 1) or len(current_function_set) == 0:
                    next_is_terminal = True  # 不添加函数节点，改为添加terminal
                    if next_is_function:
                        raise ValueError("Loop: next_is_function and next_is_terminal are both True.")
                    continue
                # 在约束函数集中选择函数节点
                function = random_state.randint(len(current_function_set))
                function = current_function_set[function]
                function = new_operator(function, random_state, self.n_features,
                                        existed_dimension, current_remaining)
                function.depth = depth  # 记录函数节点所在深度
                function.parent_distance = parent_index - len(program)  # 父节点相对于自己的距离
                if len(value_range):  # 值域不为空，传递给子节点
                    function.value_range = deepcopy(value_range)
                program.append(function)
                terminal_stack.append(function.arity)
                # 更新当前点以及其所有ancestors的total属性，所以new_operator函数中不需要再对total属性进行处理
                program = self.set_total(len(program) - 1, program)
                next_is_function = False  # 回到正常状态
            else:
                # 生成常数时父节点有value_range，则需要进行主导现象限制  'add', 'sub', 'sum', 'mean', 'max', 'min'
                if parent_name in ['mul', 'div', 'prod']:
                    if len(value_range) and not np.isinf(np.max(np.abs(value_range))) and not np.isnan(np.max(np.abs(value_range))):
                        level = np.max(np.abs(value_range))  # level > 0
                    else:
                        level = np.max(np.abs(self.variable_range))
                        # 扩大一个数量级进行常数范围扩展
                    const_range = (max(level / 10,0.1), math.ceil(level * 10))  # 常数绝对值大小在该范围内即可
                else:
                    const_range = self.variable_range
                assert const_range[1] > const_range[0]

                # 生成当前父节点最后一个子节点时要回溯至第一个sub或div节点，避免生成导致完全抵消的变量节点
                name_list = []
                index_list = []
                has_sub_div = False
                temp_parent = parent
                temp_name = parent_name
                temp_index = parent_index
                prohibit = []
                children = []
                no_cancel = False
                if terminal_stack[-1] == 1:  # 当前父节点的最后一个操作数   and parent_name != 'pow'
                    # 遍历子节点，不加入当前节点，但arity为1的函数节点会导致children为空
                    for c in program[parent_index].child_distance_list[:-1]:
                        if not isinstance(program[parent_index + c], tuple):  # children会比children2少最后一个操作数
                            no_cancel = True
                            break
                        else:  # 记录变量子节点
                            children.append(program[parent_index + c])
                    if not no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                        while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                            index_list.append(temp_index)
                            if temp_name in ['sub', 'div', 'max', 'min']:  # max和min也要避免左右子树相同
                                has_sub_div = True
                                break
                            else:  # 不是sub和div
                                name_list.append(temp_name)
                                temp_index = temp_index + temp_parent.parent_distance
                                temp_parent = program[temp_index]  # 父节点回溯
                                temp_name = temp_parent.name
                        if temp_name in ['sub', 'div', 'max', 'min'] and not has_sub_div:  # 若第一个sub或div节点是根节点
                            has_sub_div = True
                            index_list.append(temp_index)
                if has_sub_div:
                    prohibit += self.get_sub_div_prohibit(program=program,
                                                          current_index=len(program),
                                                          index_list=index_list,
                                                          name_list=name_list)
                if parent_name in ['max', 'min'] and terminal_stack[-1] == 1:  # max和min的第二个操作数
                    if isinstance(program[parent_index + 1], tuple) and program[parent_index + 1] not in prohibit:
                        prohibit.append(program[parent_index + 1])
                if parent_name == 'add' and terminal_stack[-1] == 1:  # 限制最大维度的X+X
                    # 如果是add，左子树已经是X切片向量(tuple) 
                    if isinstance(program[parent_index + 1], tuple) and program[parent_index + 1] not in prohibit:
                        # 只对最大维度的X进行 x+x限制
                        if self.calculate_dimension(program[parent_index + 1]) == self.n_features:  # 最大维度的变量节点
                            prohibit.append(program[parent_index + 1])
                            
                
                
                # 根据prohibit来生成变量节点或常数向量
                if parent_name == 'pow':
                    if terminal_stack[-1] == 2:  # pow函数的第一个操作数不接收常数
                        # 符合切片的维度
                        if existed_dimension == self.n_features or existed_dimension == self.n_features - 1:
                            terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                                vary=True)  # , prohibit=prohibit
                        else:
                            next_is_function = True
                            if next_is_terminal:
                                raise ValueError("Loop: next_is_function and next_is_terminal are both True.")
                            continue
                    else:  # pow的第二个操作数应为整数向量
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            const_int=True, prohibit=prohibit)
                # arity为1的函数节点不接收常数，其他函数节点最多只有一个常数子节点，prohibit可能导致只能生成常数节点，此时只能改为生成函数节点
                elif parent.constant_num >= min(1, parent.arity - 1):
                    if len(prohibit) and existed_dimension == self.n_features:
                        for item in prohibit:
                            # if (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                            if self.calculate_dimension(item) == self.n_features:
                                next_is_function = True
                                break
                    if existed_dimension != self.n_features and existed_dimension != self.n_features - 1:
                        next_is_function = True
                    if next_is_function:
                        if next_is_terminal:
                            self.printout(program)
                            raise ValueError("Loop: next_is_function and next_is_terminal are both True.")
                        continue
                    else:
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            vary=True, prohibit=prohibit)
                else:  # 还没有常数节点，则可以生成常数节点
                    const = False
                    if len(prohibit) and existed_dimension == self.n_features:
                        for item in prohibit:
                            if self.calculate_dimension(item) == self.n_features:
                                const = True
                                break
                    # TODO 利用existed_dimension来判断是否只生成常数节点
                    if existed_dimension != self.n_features and existed_dimension != self.n_features - 1 or const:
                        terminal = self.generate_a_terminal(random_state, existed_dimension, const=True,
                                                            const_range=const_range, prohibit=prohibit)  # 只能生成常数节点
                    else:
                        #生成切片节点
                        terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                            const_range=const_range, prohibit=prohibit)  # 可以生成切片
                if isinstance(terminal, list):  # 若生成常数节点，则需要维护constant_num属性
                    parent.constant_num += 1
                next_is_terminal = False  # 回到正常状态
                program.append(terminal)
                # 对于当前父节点函数类型的剩余arity进行更新
                terminal_stack[-1] -= 1
                # temp_range = np.array([0, 0])
                # subtree_complete = False
                # 父节点的所有arity已经被子树用光
                while terminal_stack[-1] == 0:  # 这里要对self.value_range属性进行维护
                    # subtree_complete = True
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program

                        if isinstance(program[0], _Function):
                            program[0].value_range = self.calculate_value_range(program=program,
                                                                                parent_index=0,
                                                                                parent_name=program[0].name)

                        return program
                    # 当前父节点所有子树已经完成并且弹出，此时的-1代表着当前父节点的父节点
                    # 此处表示当前父节点已经完成，那么父亲的父亲arity-1，表示他的子节点也完成了
                    terminal_stack[-1] -= 1
                    # print(program[parent_index].value_range)
                    # 计算当前父节点的valuerange，根据子树
                    program[parent_index].value_range = self.calculate_value_range(program=program,
                                                                                   parent_index=parent_index,
                                                                                   parent_name=parent_name)
                    if len(program[parent_index].value_range) == 0 or np.array_equal(program[parent_index].value_range, [0, 0]):
                        print("build_program")
                        self.printout(program)
                        print(parent_index)
                        print(program[parent_index].value_range)
                        print()
                    # 此时循环的查找当前父节点的父节点，利用while来进行返回查找（如果当前父节点已完成），返回到父亲的父亲,继续找下一个父节点
                    # 并且以此来情况terminal_stack 为新一轮生成节点获取正确的深度depth
                    parent_index = self.find_parent(parent_index, program)  # 找下一个父节点
                    parent_name = program[parent_index].name  # 父节点函数名字
                # if subtree_complete:
                #     parent.value_range = temp_range
        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        current_depth = 0
        for node in self.program:
            if isinstance(node, _Function):
                if current_depth != node.depth:
                    print("depth error: ", end='')
                assert current_depth == node.depth  # 保证深度属性不出错
                terminals.append(node.arity)
                current_depth += 1
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    current_depth -= 1
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '[' + str(node.output_dimension) + ',' + str(node.input_dimension) + ']' + '('
            else:
                if isinstance(node, tuple):  # 变量向量
                    if self.feature_names is None:
                        output += 'X[%s:%s:%s]' % (node[0], node[1], node[2])
                    else:  # 暂不修改
                        output += self.feature_names[node]
                else:  # 常数向量，但是list类型
                    output += '('
                    for num in node[0]:  # 去掉外层list
                        output += '%.3f,' % num
                    output += ')'
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    output += ')'
                    terminals[-1] -= 1
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):  # 变量分量
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:  # 常数
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:  #
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X,random_state = 42):  # X是一个由多个输入向量组成的矩阵
        # execute应具备增广切片和常数节点的能力，但在大于1维时生成的program不应该输入特征维度仅为1的X来execute
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]  # 单节点没有什么意义
        if isinstance(node, list):  # 常数向量检测，检测np.ndarray类型
            print('constant')
            return np.repeat(node[0][0], X.shape[0])  # 对每个输入向量返回一个实数
        if isinstance(node, tuple):  # 变量向量检测
            print('variable')
            return X[:, node[0]]

        apply_stack = []
        index_stack = []
        # const_to_change = []

        for index, node in enumerate(self.program):
            if isinstance(node, _Function):
                apply_stack.append([node])
                index_stack.append([index])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)
                index_stack[-1].append(index)  # 记录该子节点在program中的index
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:  # 操作数凑齐时开始计算
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = []
                for i_t, t in enumerate(apply_stack[-1][1:]):
                    if isinstance(t, list):  # 常数向量改为list[ndarray]类型，避免了后续的混淆
                        # 常数节点需要找其他变量兄弟节点来确定维度大小
                        length = 0
                        for item in apply_stack[-1][1:]:
                            if isinstance(item, tuple):  # 变量节点
                                length = X.shape[1] - item[1] - item[0]
                                # length = item[1] - item[0]
                                break
                            elif isinstance(item, np.ndarray):
                                # print(item.shape)
                                if len(item.shape) > 1:
                                    length = item.shape[0]  # dimension维度
                                else:
                                    length = 1
                                break
                        assert (length > 0)
                        if len(t[0]) < length:  # 做增广
                            level = np.max(np.abs(t[0]))
                            const_range = (max(level / 10,0.1), math.ceil(level * 10))  # 常数绝对值大小在该范围内即可

                            # level = np.max(np.abs(apply_stack[-1][0].value_range))
                            # const_range = (max(level / 10,0.1), math.ceil(level * 10))  # 常数绝对值大小在该范围内即可
                            
                            if apply_stack[-1][0].name != 'pow':
                                temp_t = self.generate_a_terminal(random_state=random_state,
                                                              output_dimension=length,
                                                              const_range=const_range,
                                                              const=True)
                            else:
                                temp_t = self.generate_a_terminal(random_state=random_state,
                                                              output_dimension=length,
                                                              const=True,
                                                              const_int=True)

                            for i, num in enumerate(t[0]):
                                temp_t[0][i] = t[0][i]
                            t = temp_t  # numpy数组是引用传递，修改t就修改了self.program中对应的常数节点

                        elif len(t[0]) > length:  # 做缩减
                            t = [t[0][:length]]  # 去掉后面多余的常数
                        # 记录要修改的常数节点，突变结束后再修改program中的对应节点
                        # const_to_change.append([index_stack[-1][i_t + 1], t])  # 第一个index是父节点的index
                        # print(f'const1:{np.array(t).shape}')
                        temp = np.repeat(t, X.shape[0], axis=0)  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        # print(f'const3:{temp.shape}', end='\n\n')
                        terminals.append(temp)
                    elif isinstance(t, tuple):
                        # t[1]表示从右开始数到切片末尾所需的次数
                        temp = X[:, t[0]:X.shape[1] - t[1]:1]  # n_samples x dimension of t
                        # 调整维度顺序，n_samples调整为最后一维，因为sum和prod等aggregate函数是按axis=0来进行计算的
                        # print(f'variable1:{temp.shape}')
                        temp = np.transpose(temp, axes=(1, 0))  # dimension x n_samples
                        # print(f'variable2:{temp.shape}', end='\n\n')
                        terminals.append(temp)
                    else:  # 中间结果，即np.ndarray类型，无需额外处理
                        terminals.append(t)  # arity x dimension x n_samples
                # 聚集函数要保证不在样本数维度上做聚集计算，arity>1时在各个操作数维度上进行聚集计算，arity=1时在特征数维度上进行聚集计算
                if function.name in ['sum', 'prod', 'mean']:
                    terminals = np.array(terminals)
                    # arity>1时sum和prod保持输入和输出维度相同，arity=1时输入为向量，输出为实数
                    if terminals.ndim > 2 and terminals.shape[0] == 1:
                        # arity为1时去掉操作数维度，输出结果会少一个维度，此时要统一格式，将增加大小为1的特征数维度
                        intermediate_result = function(terminals[0])
                        intermediate_result = intermediate_result.reshape(1, -1)
                        # print(f'aggregate1: {intermediate_result.shape}')
                    else:  # arity>1时与其他函数输出结果的shape相同
                        intermediate_result = function(terminals)
                        # print(f'aggregate2: {intermediate_result.shape}')
                else:
                    intermediate_result = function(*terminals)
                # print(f"{apply_stack[-1][0].name} result : {intermediate_result}")
                
                
                    # print(f'others: {intermediate_result.shape}')
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    index_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                    index_stack[-1].append(0)
                else:
                    # 做缩减之后无需修改原program的list
                    # 因为我们需要在线计算，修改了会出问题
                    # for item in const_to_change:
                    #     self.program[item[0]] = item[1]  # 修改对应常数节点
                    return intermediate_result[0]  # 最后去掉特征数维度，只保留样本数维度
        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)
        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]
        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def check_y_legal(self,y_pred):
        # 判断生成的y是否有非法的objective value
        if (np.isnan(y_pred).any() or np.isinf(y_pred).any()):
            # print("-----非法目标值函数-----")
            # printout(self.program,max_dim=X.shape[-1])
            # print_formula(self.program, show_operand=True,max_dim=X.shape[-1])
            return False
        # 此处做个判断，如果生成的是常数函数(y的最大最小值一样) 
        # 换一个方法 判断生成的函数的y的方差
        if np.max(y_pred) - np.min(y_pred) <= 1e-8 or np.max(y_pred) == np.min(y_pred) :
            # print("-----常数函数-----")
            # print(np.max(y_pred),np.min(y_pred))
            # printout(self.program,max_dim=X.shape[-1])
            # print_formula(self.program, show_operand=True,max_dim=X.shape[-1])
            return False
        return True
        
    def check_func_legal(self):
        num_sample = 1000
        X_2D = np.array(create_initial_sample(2, n=num_sample*2, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=100))
        X_5D = np.array(create_initial_sample(5, n=num_sample*5, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=100))
        X_10D = np.array(create_initial_sample(10, n=num_sample*10, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=100))
        y_2D , y_5D ,y_10D= self.execute(X_2D, random_state=100) ,  self.execute(X_5D, random_state=100)  ,self.execute(X_10D, random_state=100)  
        total_y = [y_2D,y_5D,y_10D]
        for y_pred in total_y:
        # 判断生成的y是否有非法的objective value
            if (np.isnan(y_pred).any() or np.isinf(y_pred).any() or np.sum(np.abs(y_pred) > 1e50)):
                # print("-----非法目标值函数-----")
                # printout(self.program,max_dim=X.shape[-1])
                # print_formula(self.program, show_operand=True,max_dim=X.shape[-1])
                return False
        return True


    def raw_fitness(self, X, y, sample_weight, random_state):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        # 问题的y值用于计算ela
        # 30D的原始计算
        y_pred = self.execute(X, random_state=random_state)
        
        self.coordi_2D = None
        X_2D = np.array(create_initial_sample(2, n=250*2, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=100))
        X_5D = np.array(create_initial_sample(5, n=250*5, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=100))
        y_2D , y_5D = self.execute(X_2D, random_state=random_state) ,  self.execute(X_5D, random_state=random_state)  
        dim_list = [2,5,10]
        num_dim = len(dim_list)
        
        penalty = 1e2
        total_x = [X_2D,X_5D,X]
        total_y = [y_2D,y_5D,y_pred]
        total_fitness = np.full((num_dim),penalty)
        not_cal_ela = np.zeros((num_dim))
        gp_ela_pure_list = []
        coordi_2D_list = []

        for dim,y in enumerate(total_y):
            if not self.check_y_legal(y) or not self.check_func_legal():
                # y非法，则无需计算ela
                not_cal_ela[dim] = 1
                # 同时设置fitness是penalty(初始化时已是)
        # 根据not_cal_ela，计算ela
        for dim in range(num_dim):
            if not not_cal_ela[dim]:
                # 计算该individual的ela
                ub = 5
                random_seed = 100
                try:
                    gp_ela_pure,_,_ = get_ela_feature(GP_problem(self.execute,self.problemID,lb = -ub,ub = ub,dim = total_x[dim].shape[-1],random_state=random_state),\
                        total_x[dim],total_y[dim],random_state = random_seed)
                    gp_ela_pure_list.append(gp_ela_pure)
                except:
                    not_cal_ela[dim] = 1
                    gp_ela_pure_list.append(None)
                    continue
            else:
                # 在该维度不计算ela
                gp_ela_pure_list.append(None)
            

        # 如果各个维度的ela都无法计算，直接penalty
        if not_cal_ela.sum() >= num_dim:
            return penalty
        else:
            # 至少有一个可以算，根据not_cal_ela来判断
            for dim in range(num_dim):
                if not not_cal_ela[dim]:
                    tmp_ela = self.scaler.transform(gp_ela_pure_list[dim].reshape(1,-1))
                    gp_problem_2D = get_encoded(self.model,tmp_ela)
                    # 降维后出现了nan 
                    if np.isnan(gp_problem_2D).any() or np.isinf(gp_problem_2D).any():
                        # print("-----降维后出现NaN or Inf-----")
                        # 设置不可计算not_cal_ela
                        not_cal_ela[dim] = 1
                        # 设置一个外面的坐标
                        coordi_2D_list.append(np.full((2),100))
                    else:
                        # 计算fitness,更新
                        total_fitness[dim] = self.metric(gp_problem_2D,self.problem_coord,sample_weight)
                        coordi_2D_list.append(gp_problem_2D)
                else:
                    # 对于不计算ela的维度
                    # 直接分配一个外面的坐标即可
                    coordi_2D_list.append(np.full((2),100))
        # 返回最小的fitness对应的维度                
        dim = np.argmin(total_fitness)
        raw_fitness = total_fitness[dim]
        # 存储获得的该function对应的ela特征
        self.ela_feature = gp_ela_pure_list[dim]
        self.ela_feature_list = gp_ela_pure_list
        # 存储降维后的坐标
        self.coordi_2D = coordi_2D_list[dim]
        self.coordi_2D_list = coordi_2D_list
        # 存储best dim对应的函数维度
        self.best_dim = dim_list[dim]

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    # 根据remaining，constant_num和parent_name来筛选可交换节点
    def get_random_subtree(self, random_state, program=None, output_dimension=1,
                    remaining=None, constant_num=0, prohibit=None, no_root=False, min_depth=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.
        output_dimension : int
            The dimension of subtree's output
        remaining: list
        constant_num: int
        prohibit: list
        no_root: bool
        min_depth: int
        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        indices = []  # 记录(输出)维度等于output_dimension的节点的下标
        for index, node in enumerate(program):
            if output_dimension == self.calculate_dimension(node):  # 满足维度相同要求
                if isinstance(node, _Function):  # 是函数节点，则需要检查是否有remaining要求以及是否有连续嵌套限制
                    if remaining is None:
                        if prohibit is None or len(prohibit) == 0 or node.name not in prohibit:
                            if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                                indices.append(index)
                    elif self.subtree_state_larger(remaining, node.total):  # 有remaining要求
                        if prohibit is None or len(prohibit) == 0 or node.name not in prohibit:
                            if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                                indices.append(index)
                elif isinstance(node, list):  # 是常数节点，则需要检查是否满足constant_num的约束
                    if constant_num == 0:  # 可以添加常数节点
                        if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                            indices.append(index)
                else:  # 变量节点
                    if min_depth is None or self.get_depth(index=index, program=program) >= min_depth:
                        indices.append(index)
        if no_root and 0 in indices:
            indices.remove(0)  # 去除根节点坐标
        if len(indices) == 0:  # 加入remaining后可能会导致没有可交叉部分
            return -1, -1  # -1, -1表示没有符合要求的子树
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(program[index], _Function) else 0.1 for index in indices])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
        start = indices[start]
        # start = indices[random_state.randint(len(indices))]  # 随机挑选其中一个节点的索引作为起点
        if start != 0:  # 不是根节点parent_index才有意义
            parent_index = self.find_parent(start, program)  # 寻找父节点函数
            # 父节点是pow函数且子树根节点不是pow的第一操作数
            if program[parent_index].name == 'pow' and start != parent_index + 1:
                if remaining is None:  # 仅没有remaining要求可以如此更换
                    # 将子树根节点改为pow函数或其第一个操作数，若父节点pow函数是根节点，则不可以选中
                    if (no_root and parent_index == 0) or random_state.randint(0, 2) == 0:
                        start = parent_index + 1
                    else:
                        start = parent_index
                else:
                    return -1, -1  # -1, -1表示没有符合要求的子树
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1
        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return deepcopy(self.program)

    def crossover(self, donor, random_state):  # 遍历所有可能的维度大小
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        ###
        # 求两个公式所有输出维度的交集
        common_dimensions = set(self.get_output_dimensions()).intersection(set(self.get_output_dimensions(donor)))
        if len(common_dimensions) == 0:  # 交集为空。这种情况理论上不存在，但若交集只有1，那可能只是完全交换
            raise ValueError('Crossover: The intersection of output_dimensions of two trees is empty.')
        # 生成一个不重复随机索引数列
        index_list = random_state.permutation(range(len(common_dimensions)))
        prohibit = []  # 用于限制donor的子树选择
        counter = 0
        for index in index_list:  # 遍历所有可能的输出维度
            dimension = 0
            for i, item in enumerate(common_dimensions):
                if i == index:
                    dimension = item
                    # 2024.11.4
                    break
            # start, end = self.get_random_subtree(random_state, output_dimension=dimension, min_depth=2)  # 前两层节点不参与突变
            
            # 随机寻找相同维度输出的子树
            start, end = self.get_random_subtree(random_state, output_dimension=dimension, no_root=True)  # 根节点不参与突变
            removed = range(start, end)
            
            # 寻找到了之后，找这个子树的父节点
            if start > 0:  # 不是根节点，说明父节点是函数节点，然后父节点的remaining和constant_num才有意义
                parent_index = self.find_parent(start)  # 需要找到start的父节点的remaining
                parent_name = self.program[parent_index].name
                remaining = self.program[parent_index].remaining
                if self.program[parent_index].arity == 1 or parent_name == 'pow':
                    constant_num = 1  # arity为1的函数节点以及pow函数节点不接收常数
                    # 设置为1，意思是不能再生成常数
                else:
                    constant_num = self.program[parent_index].constant_num
            elif start == 0:  # start和parent_index相同，即start=0，为根节点，则没有要求
                parent_index = 0
                parent_name = None
                remaining = deepcopy(default_remaining)
                constant_num = 0
            else:  # start小于0，即只有根节点输出维度为1且被no_root避免导致get_random_subtree返回(-1, -1)，这种情况直接重新找另一个维度
                continue
            
            # 直接在此处可以根据选出要被替换子树的父节点的性质，提前加入prohibit集
            if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
                # 此处其实应该把pow也加进去？不然会导致变化太大了？但是如果直接在remain设置了应该也不会出现嵌套
                if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt','mean']:
                    prohibit.append(parent_name)  # 防止这些函数连续嵌套
                elif parent_name == 'exp':  # 防止exp和log连续嵌套
                    prohibit.append('log')
                elif parent_name == 'log':
                    prohibit.append('exp')
                    prohibit.append('neg')
                # 下面是避免冗余操作
                elif parent_name == 'sqrt' :
                    if 'abs' not in prohibit:
                        prohibit.append('abs')
                    if 'neg' not in prohibit:
                        prohibit.append('neg')
                elif parent_name == 'abs' and 'neg' not in prohibit:
                    prohibit.append('neg')
                    
            
            # 随机选择donor中一个指定输出维度子树，且子树状态兼容，且避免交换后同时存在两个常数子节点
            # 此处加入原子树父节点的remaining和constant的限制信息，寻找符合要求的子树，并且已经给出了基本的prohibit嵌套限制
            donor_start, donor_end = self.get_random_subtree(random_state, program=donor, output_dimension=dimension,
                                                             remaining=remaining, constant_num=constant_num,
                                                             prohibit=prohibit)
            if (donor_start, donor_end) == (-1, -1):  # 没找到符合要求的子树
                continue
            if start != 0 and parent_name == 'add':
                if start != parent_index + 1:  # 不是左子节点
                    # 那么左子树的index就是这个
                    another = parent_index + 1
                else:
                    # 否则当前start是左子节点，右子树的index根据父亲的child distance的最后一个确定
                    another = parent_index + self.program[parent_index].child_distance_list[-1]
                # add的两个子节点是变量节点且维度相同
                # 要替换过来的donor是变量，并且对于add而言，被替换子树（变量）的兄弟也是同dim的变量
                if isinstance(self.program[another], tuple) and self.program[another] == donor[donor_start]:
                    if self.calculate_dimension(self.program[another]) == self.n_features:
                        continue  # 维度是最大维度则重新突变，即出现最大维度的变量节点X+X
            init_depth = self.get_depth(index=start)  # 求self.program子树根节点的深度
            replacement = self.set_depth(init_depth=init_depth, program=donor[donor_start:donor_end])  # 设置donor子树列表的深度
            if isinstance(replacement[0], _Function):  # 根节点是函数节点则更新remaining
                remaining = self.update_remaining(remaining, replacement[0])
                replacement[0].parent_distance = parent_index - start  # donor子树根节点的parent_distance属性也要更新
            # 对replacement整个子树，根据上面得到的根节点的新的remaining，来更新他的子节点的remaining以及根节点自身的remaining
            replacement = self.set_remaining(remaining=remaining, program=replacement)  # 设置replacement的remaining属性
            donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))# 表示donor没捐出去的那部分index
            assert self.calculate_dimension(replacement[0]) == self.calculate_dimension(self.program[start])  # 保证两者维度一致

            # 更新父节点和子节点相对距离
            program = deepcopy(self.program)  # 使用deepcopy防止共用错误
            # offset实际上就是替换进去的子树和被替换子树的长度差
            # 因为如果list里面替换掉一段内容subtree，那么在子树后面的节点，他们记录的parent distance会因为自己的parent被换走，新子树导致距离变化
            offset = (donor_end - donor_start) - (end - start)
            if offset != 0:
                if start != 0:
                    fence = 0  # 从根节点开始更新parent_distance和child_distance_list属性
                    while fence < start:
                        for i, distance in enumerate(program[fence].child_distance_list[::-1]):
                            if fence + distance > start:  # 位于start后的子节点
                                # 该子节点是函数节点，则更新parent_distance
                                if isinstance(program[fence + distance], _Function):
                                    program[fence + distance].parent_distance -= offset
                                program[fence].child_distance_list[- 1 - i] += offset
                            else:  # fence + distance <= start
                                fence = fence + distance
                                break

            # 设置total值
            subtree = program[start: end]
            for i, node in enumerate(subtree):  # 遍历subtree，减去其中函数节点的total值，删除这部分子树带来的total权重（向上一直对父节点进行删除total）
                program = self.set_total(index=start + i, program=program, subtract=True)
            for i, node in enumerate(replacement):  # 遍历replacement，其中函数节点的total值均归零
                replacement = self.set_total(index=i, program=replacement, subtract=True)
            new_program = program[:start] + replacement + program[end:]
            for i, node in enumerate(replacement):  # 遍历replacement，加上其中函数节点的total值
                new_program = self.set_total(index=start + i, program=new_program)
            # 对突变后的新子树进行sub和div检查，具体方法是对交换过来的新子树的每个变量节点进行父节点回溯，找start以上的第一个sub和div节点
            cancel = False
            has_exp = False
            temp_index = self.find_parent(start, new_program)
            temp_parent = new_program[temp_index]
            while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                if temp_parent.name == 'exp' and temp_parent.depth < init_depth:
                    has_exp = True
                    break
                else:  # 不是sub和div
                    temp_index = temp_index + temp_parent.parent_distance
                    temp_parent = new_program[temp_index]  # 父节点回溯
            # 此时已回溯到根节点，对根节点进行判断
            if temp_parent.name == 'exp' and temp_parent.depth < init_depth and not has_exp:
                has_exp = True
            if has_exp:  # 有exp节点则需要检查突变子树中有无sum节点
                for i, node in enumerate(replacement):  # 遍历replacement，加上其中函数节点的total值
                    if isinstance(node, _Function) and node.name == 'sum':
                        cancel = True
                        break
            if not cancel:
                for i, item in enumerate(new_program[start:start + len(replacement)]):  # 遍历replacement部分
                    # print(start + i)
                    if isinstance(item, tuple):  # 遍历新子树的所有变量节点来检查sub和div抵消
                        # 仅parent_index的最后一个变量子节点
                        name_list = []
                        index_list = []
                        has_sub_div = False
                        temp_index = self.find_parent(start + i, new_program)
                        temp_parent = new_program[temp_index]
                        temp_name = temp_parent.name
                        while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                            index_list.append(temp_index)
                            if temp_name in ['sub', 'div', 'max', 'min'] and temp_parent.depth < init_depth:
                                has_sub_div = True
                                break
                            else:  # 不是sub和div
                                name_list.append(temp_name)
                                temp_index = temp_index + temp_parent.parent_distance
                                temp_parent = new_program[temp_index]  # 父节点回溯
                                temp_name = temp_parent.name
                        # 若第一个sub或div节点是根节点
                        if temp_name in ['sub', 'div', 'max',
                                         'min'] and temp_parent.depth < init_depth and not has_sub_div:
                            has_sub_div = True
                            index_list.append(temp_index)
                        if has_sub_div:  # 有start以上的sub和div祖先节点，则检查其另一子树有无相同子支，但这样会导致左右重复检测(组合变排列)
                            left = index_list[-1] + 1
                            right = index_list[-1] + new_program[index_list[-1]].child_distance_list[-1]
                            root_div = new_program[index_list[-1]].name == 'div'
                            cancel = self.check_subtree_identity(root_div=root_div,
                                                                 left_subtree=self.get_subtree(root=left,
                                                                                               program=new_program),
                                                                 right_subtree=self.get_subtree(root=right,
                                                                                                program=new_program))
                        if cancel:  # 存在抵消不需要检测剩下的节点，直接重新突变
                            break
            # 有sub和div的抵消或突变后树的深度不满足要求，则重新突变
            if cancel or self.mutate_depth is not None and \
                    not (self.mutate_depth[0] < self.get_max_depth(new_program) < self.mutate_depth[1]):
                counter += 1
                if counter >= 6:  # 尝试次数超过5次就停止，不交叉突变
                    return deepcopy(self.program), 0, 0
                continue
            # y_pred = self.execute_test(new_program, X_train, random_state)
            # if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
            #     print("-----常数函数(crossover)-----")
            #     self.printout(self.program)
            #     self.printout(replacement)
            #     self.printout(new_program)
            #     self.print_formula(new_program, show_operand=True)
            temp_constant_num = 0
            for i in new_program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                if isinstance(new_program[parent_index + i], list):
                    temp_constant_num += 1
            new_program[parent_index].constant_num = temp_constant_num
            return new_program, removed, donor_removed
        return deepcopy(self.program), 0, 0

    def subtree_mutation(self, random_state):  # 子树突变
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # 挑选一个随机输出维度子树
        output_dimensions = self.get_output_dimensions()
        # 生成一个不重复随机索引数列
        index_list = random_state.permutation(range(len(output_dimensions)))
        prohibit = []
        counter = 0
        for index in index_list:
            output_dimension = 0
            for i, item in enumerate(output_dimensions):
                if i == index:
                    output_dimension = item
                    break
            # hoist突变可以对整个树进行裁剪[，前两层节点不参与突变]
            start, end = self.get_random_subtree(random_state, output_dimension=output_dimension)  # , min_depth=2
            # if (start, end) != (-1, -1):
            #     break
            subtree = self.program[start:end]
            if start > 0:  # 不是根节点，说明父节点是函数节点，然后父节点的remaining和constant_num才有意义
                parent_index = self.find_parent(start)  # 需要找到start的父节点的remaining
                parent_name = self.program[parent_index].name
                remaining = self.program[parent_index].remaining
                if self.program[parent_index].arity == 1 or parent_name == 'pow':
                    constant_num = 1  # arity为1的函数节点以及pow函数节点不接收常数
                else:
                    constant_num = self.program[parent_index].constant_num
            elif start == 0:  # start和parent_index相同，即start=0，为根节点，则没有要求
                parent_index = 0
                parent_name = None
                remaining = deepcopy(default_remaining)
                constant_num = 0  # 0表示可以生成常数
            else:  # start小于0，即只有根节点输出维度为1且被no_root避免导致get_random_subtree返回(-1, -1)，这种情况直接重新找另一个维度
                continue
            if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
                if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt','mean']:
                    prohibit.append(parent_name)  # 防止这些函数连续嵌套
                elif parent_name == 'exp':  # 防止exp和log连续嵌套
                    prohibit.append('log')
                elif parent_name == 'log':
                    prohibit.append('exp')
                    prohibit.append('neg')
                # 下面是避免冗余操作
                elif parent_name == 'sqrt' :
                    if 'abs' not in prohibit:
                        prohibit.append('abs')
                    if 'neg' not in prohibit:
                        prohibit.append('neg')
                elif parent_name == 'abs' and 'neg' not in prohibit:
                    prohibit.append('neg')
            # hoist突变前面和crossover一样，先找出一个子树的start和end，并且获得他的父亲节点信息
            # 然后第二步实际上是从这个从自身找出的子树，再提取出一个小的子树（这样就可以理解为缩减了原本的子树）
            # 原来的crossover则是从donor树里面找这样的子树
            sub_start, sub_end = self.get_random_subtree(random_state, program=subtree,
                                                         output_dimension=self.calculate_dimension(self.program[start]),
                                                         remaining=remaining, constant_num=constant_num,
                                                         prohibit=prohibit, no_root=True)  # 不能选择根节点，否则没有变化
            if (sub_start, sub_end) == (-1, -1):  # 没找到符合要求的子树
                continue
            if start != 0 and parent_name == 'add':
                if start != parent_index + 1:  # 不是左子节点
                    another = parent_index + 1
                else:
                    another = parent_index + self.program[parent_index].child_distance_list[-1]
                # add的两个子节点是变量节点且维度相同
                # 第一个条件是要被缩减的子树的兄弟节点another，如果是变量节点
                # 第二个条件是如果这个节点和 被缩减子树选出来最后提取出来的小子树的根节点（头节点）一样
                # 第三个条件是如果是全维度的变量节点，那么就触发了条件，需要更换维度重新突变
                if isinstance(self.program[another], tuple) and self.program[another] == subtree[sub_start]:
                    if self.calculate_dimension(self.program[another]) == self.n_features:
                        continue  # 维度是最大维度则重新突变，即出现最大维度的变量节点X+X
            # 需要对缩减后的子树重新设置depth
            # 先得到原始未缩减子树的深度，然后修改hoist各点的depth
            init_depth = self.get_depth(start, self.program)
            hoist = self.set_depth(init_depth, subtree[sub_start:sub_end])  # 设置hoist各点的depth属性
            # 并且需要更新hoist点的remaining（根据被hoist子树的父亲的remain来对应的设置根节点的remaining）
            if isinstance(hoist[0], _Function):  # 根节点是函数节点则更新remaining
                remaining = self.update_remaining(remaining, hoist[0])
                hoist[0].parent_distance = parent_index - start  # hoist子树根节点的parent_distance属性也要更新
            hoist = self.set_remaining(remaining, hoist)  # 设置hoist各点的remaining属性

            # 更新父节点和子节点相对距离
            program = deepcopy(self.program)
            offset = (sub_end - sub_start) - (end - start)
            if offset != 0:
                if start != 0:
                    fence = 0  # 从根节点开始
                    while fence < start:
                        for i, distance in enumerate(program[fence].child_distance_list[::-1]):
                            if fence + distance > start:  # 位于start后的子节点
                                if isinstance(program[fence + distance], _Function):  # 该子节点是函数节点，则更新parent_distance
                                    program[fence + distance].parent_distance -= offset
                                program[fence].child_distance_list[- 1 - i] += offset
                            else:  # fence + distance <= start
                                fence = fence + distance
                                break

            # 修改被hoist的部分的ancestors的total值
            for i, node in enumerate(subtree):  # 遍历subtree，减去其中函数节点的total值
                program = self.set_total(index=start + i, program=program, subtract=True)  # 对program变量逐步更新
            for i, node in enumerate(hoist):  # 遍历hoist，减去其中函数节点的total值
                hoist = self.set_total(index=i, program=hoist, subtract=True)  # hoist的total值归零
            new_program = program[:start] + hoist + program[end:]
            for i, node in enumerate(hoist):  # 遍历hoist，加上其中函数节点的total值
                new_program = self.set_total(index=start + i, program=new_program)
            # 对突变后的新子树进行sub和div检查，具体方法是对交换过来的新子树的每个变量节点进行父节点回溯，找start以上的第一个sub和div节点
            cancel = False
            # hoist突变不会产生新结构，不需要检测exp套sum节点
            for i, item in enumerate(new_program[start:start + len(hoist)]):  # 遍历新子树的所有变量节点
                # print(start + i)
                if isinstance(item, tuple):
                    name_list = []
                    index_list = []
                    has_sub_div = False
                    temp_index = self.find_parent(start + i, new_program)
                    temp_parent = new_program[temp_index]
                    temp_name = temp_parent.name
                    while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                        index_list.append(temp_index)
                        # temp_parent.depth < init_depth 是因为需要找到，对于hoist新子树放进去之后，上层的父节点会不会突然出现左右子树抵消的结构
                        if temp_name in ['sub', 'div', 'max', 'min'] and temp_parent.depth < init_depth:
                            has_sub_div = True
                            break
                        else:  # 不是sub和div
                            name_list.append(temp_name)
                            temp_index = temp_index + temp_parent.parent_distance
                            temp_parent = new_program[temp_index]  # 父节点回溯
                            temp_name = temp_parent.name
                    # 若第一个sub或div节点是根节点
                    if temp_name in ['sub', 'div', 'max', 'min'] and temp_parent.depth < init_depth and not has_sub_div:
                        has_sub_div = True
                        index_list.append(temp_index)
                    if has_sub_div:  # 有start以上的sub和div祖先节点，则检查其另一子树有无相同子支，但这样会导致左右重复检测(组合变排列)
                        left = index_list[-1] + 1
                        right = index_list[-1] + new_program[index_list[-1]].child_distance_list[-1]
                        root_div = new_program[index_list[-1]].name == 'div'
                        cancel = self.check_subtree_identity(root_div=root_div,
                                                             left_subtree=self.get_subtree(root=left,
                                                                                           program=new_program),
                                                             right_subtree=self.get_subtree(root=right,
                                                                                            program=new_program))
                    if cancel:  # 存在抵消不需要检测剩下的节点，直接重新突变
                        break
            # 有sub和div的抵消或突变后树的深度不满足要求，则重新突变
            # self.printout(new_program)
            if cancel or self.mutate_depth is not None and \
                    not (self.mutate_depth[0] < self.get_max_depth(new_program) < self.mutate_depth[1]):
                counter += 1
                if counter >= 6:  # 尝试次数超过5次就停止，不进行突变
                    return deepcopy(self.program), 0
                continue
            # Determine which nodes were removed for plotting
            removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))

            # 重新统计hoist之后，父节点的常数子节点的情况
            temp_constant_num = 0
            for i in new_program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                if isinstance(new_program[parent_index + i], list):
                    temp_constant_num += 1
            new_program[parent_index].constant_num = temp_constant_num
            return new_program, removed
        return deepcopy(self.program), 0

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # program = copy(self.program)
        program = deepcopy(self.program)

        # Get the nodes to modify
        # 对program的每个节点，都得到一个random的概率，如果概率小于点突变的概率
        # 那就列入备选突变中
        # 得到备选突变节点的idx
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]  # 返回满足条件的索引数组
        for node in mutate:  # 突变节点索引列表
            if isinstance(program[node], _Function):
                if program[node].name in ['sum', 'prod', 'pow', 'mean']:  # 不能与pow互换，因为会导致nan值的出现
                    continue  # 这几个函数节点暂时不进行点突变，sum节点不突变，只需要在突变出exp节点后检测子树中是否有sum节点
                else:
                    # 此处是获得要点突变node的父节点信息
                    if node != 0:
                        parent_index = self.find_parent(node, program)
                        remaining = program[parent_index].remaining  # 根据父节点的remaining来约束突变范围
                        parent_name = program[parent_index].name
                    else:
                        remaining = deepcopy(default_remaining)
                        parent_name = None
                    # 根据父节点的remaining以及parent_name(限制嵌套)和当前节点的arity
                    # 选取可以突变的function_set
                    function_set = self.clip_function_set(remaining, self.arities[program[node].arity], no_pow=True,
                                                          parent_name=parent_name, total=program[node].total)
                    if len(function_set) != 0:  # 有可突变的函数才突变
                        origin = program[node]
                        # 随机抽取里面的function
                        replacement = len(function_set)
                        replacement = random_state.randint(replacement)
                        replacement = function_set[replacement]
                        replacement = new_operator(replacement, random_state,
                                                   self.n_features, origin.output_dimension,
                                                   remaining)  # 更换合适的运算符，当前节点的total值同样需要维护
                        # 要继承depth，parent_distance，child_distance_list等所有属性
                        replacement.depth = origin.depth
                        replacement.value_range = deepcopy(origin.value_range)
                        replacement.parent_distance = origin.parent_distance
                        replacement.child_distance_list = deepcopy(origin.child_distance_list)
                        replacement.constant_num = origin.constant_num

                        # 将origin替换为replacement，还需要更新ancestors的total值
                        program = self.set_total(node, program, subtract=True)  # 减去当前点以及其所有ancestors的原先函数的total值
                        replacement.total = deepcopy(origin.total)
                        program[node] = replacement
                        program = self.set_total(node, program, subtract=False)  # 然后再加上新的replacement的total增值
                        # 根据当前点突变父节点的remaining，和当前突变完的节点，来更新当前节点的remaining
                        remaining = self.update_remaining(remaining, program[node])
                        # 根据当前节点的remaining，更新下面所有节点的remaing
                        program[node:] = self.set_remaining(remaining, program[node:])

                        # 下面要判断是否点突变后，会出现我们的条件限制
                        # 比如x+x，sub和div抵消，exp后面出现sum
                        
                        cancel = False
                        if replacement.name == 'add':  # 突变为add函数节点，则判断两个子节点是否都是最大维度的变量节点
                            left_child_index, right_child_index = node + 1, node + program[node].child_distance_list[-1]
                            if isinstance(program[left_child_index], tuple) and \
                                    program[left_child_index] == program[right_child_index]:
                                if self.calculate_dimension(program[left_child_index]) == self.n_features:
                                    cancel = True
                        elif replacement.name == 'exp':  # 突变为exp函数节点，则判断其子树有无sum节点
                            subtree = self.get_subtree(node + 1, program)  # 直接子节点为子树的根节点
                            for item in subtree:
                                if isinstance(item, _Function) and item.name == 'sum':
                                    cancel = True
                                    break
                        if not cancel:
                            if replacement.name in ['sub', 'div', 'max', 'min']:  # 点突变如果突变出sub或者div，也要避免抵消
                                # 检查sub或者div的左右子树是否相同即可
                                left_child_index, right_child_index = node + 1, node + \
                                                                      program[node].child_distance_list[-1]
                                root_div = replacement.name == 'div'
                                cancel = self.check_subtree_identity(root_div=root_div,
                                                                     left_subtree=self.get_subtree(
                                                                         root=left_child_index,
                                                                         program=program),
                                                                     right_subtree=self.get_subtree(
                                                                         root=right_child_index,
                                                                         program=program))
                            elif node != 0:  # 非根节点突变出其他的函数也要回溯父节点进行sub和div抵消检测
                                sub_div_index = 0
                                has_sub_div = False
                                # 不断寻找当前节点的父节点，往上查找是否存在sub或者div
                                parent_index = self.find_parent(node, program)
                                temp_index = parent_index
                                temp_parent = program[temp_index]
                                temp_name = temp_parent.name
                                while temp_parent.parent_distance != 0:  # 一直回溯到根节点
                                    if temp_name in ['sub', 'div', 'max', 'min']:
                                        sub_div_index = temp_index
                                        has_sub_div = True
                                        break
                                    else:  # 不是sub和div
                                        temp_index = temp_index + temp_parent.parent_distance
                                        temp_parent = program[temp_index]  # 父节点回溯
                                        temp_name = temp_parent.name
                                if temp_name in ['sub', 'div', 'max', 'min'] and not has_sub_div:  # 若第一个sub或div节点是根节点
                                    has_sub_div = True
                                    sub_div_index = temp_index
                                if has_sub_div:
                                    # 如果向上查找到了sub或者div，那么就递归的检查他的左右子树即可
                                    left_child_index = sub_div_index + 1
                                    right_child_index = sub_div_index + program[sub_div_index].child_distance_list[-1]
                                    root_div = program[sub_div_index].name == 'div'
                                    cancel = self.check_subtree_identity(root_div=root_div,
                                                                         left_subtree=self.get_subtree(
                                                                             root=left_child_index,
                                                                             program=program),
                                                                         right_subtree=self.get_subtree(
                                                                             root=right_child_index,
                                                                             program=program))
                        if cancel:  # 如果出现了抵消，则替换回原来的节点
                            program = self.set_total(node, program, subtract=True)  # 减去当前点以及其所有ancestors的原先函数的total值
                            origin.total = deepcopy(replacement.total)
                            program[node] = origin
                            program = self.set_total(node, program, subtract=False)  # 然后再加上origin的total增值
                            remaining = self.update_remaining(remaining, program[node])
                            program[node:] = self.set_remaining(remaining, program[node:])
            else:  # 变量向量或常数向量，要加入对sub和div抵消的检测，若抵消则重新突变
                existed_dimension = self.calculate_dimension(program[node])  # 计算该点的维度
                parent_index = self.find_parent(node, program)  # 找父节点,父节点名字
                parent_name = program[parent_index].name

                # 从下至上存放着不是sub或者div的那些父节点
                name_list = []
                # 从下至上存放着当前节点的所有父节点（直到搜到div或者sub为止，并且包含）
                index_list = []
                has_sub_div = False
                no_cancel = False
                temp_index = parent_index
                temp_name = parent_name
                temp_parent = program[parent_index]
                prohibit = []
                # 一直回溯到根节点
                while temp_parent.parent_distance != 0:
                    index_list.append(temp_index)
                    if temp_name in ['sub', 'div', 'max', 'min']:
                        has_sub_div = True
                        break
                    else:  # 不是sub和div
                        name_list.append(temp_name)
                        temp_index = temp_index + temp_parent.parent_distance
                        temp_parent = program[temp_index]  # 父节点回溯
                        temp_name = temp_parent.name
                # 若第一个sub或div节点是根节点
                if temp_name in ['sub', 'div', 'max', 'min'] and not has_sub_div:
                    has_sub_div = True
                    index_list.append(temp_index)
                if has_sub_div:
                    # 有sub和div祖先节点且当前点位于其右子树
                    # 第一个条件是index_list为1：父亲节点就是sub或者div 并且 node 不是父节点的左子树
                    # 第二个条件是 sub或者div在上面几层，并且sub或者div连接着这个node的几个父亲节点，不是位于左子树
                    if len(index_list) == 1 and node != index_list[-1] + 1 or \
                            len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:
                        temp_index = index_list[-1] + 1  # 从左子节点开始
                    else:  # 当前点位于左子树
                        # 从sub或者div的右子节点开始
                        temp_index = index_list[-1] + program[index_list[-1]].child_distance_list[-1]  
                    if program[index_list[-1]].name == 'div':
                        # 如果node的父亲节点不是div(只有这种情况name_list会有长度)
                        if len(name_list):
                            complete = False
                            name_index = 1
                            # name_list[- name_index]刚开始是div的直接子节点，
                            while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                # 如果此时div的直接子节点是abs和neg
                                # 由于div的性质，这些不影响抵消，所以可以在判断中跳过
                                name_index += 1
                                # 这个判断的意思是，如果跳过了abs或者neg之后，已经到达了当前进行点突变的点
                                # 因为name_list是存放不包括当前点和div/sub点的中间那些父亲节点
                                # -name_index相当于是从上往下遍历这个name_list
                                # 但是理论上这个while循环最多进行一次，因为name_index +=1之后跳到下面的父节点
                                # 而且abs和neg不会嵌套
                                # 所以如果不满足下面if条件，那么这段代码的含义，只是让name_index跳到下一个非abs和neg的父节点
                                # 所以这段while循环是专门处理，当前节点和div节点，中间只相隔了一个abs和neg的情况
                                if name_index > len(name_list):  # 等价于右边name_list为空
                                    # 如果此时另一支的节点直接是变量节点，那么prohibit就避免当前node突变成变量节点
                                    if isinstance(program[temp_index], tuple):
                                        prohibit.append(program[temp_index])
                                    elif isinstance(program[temp_index], _Function):
                                        # 如果此时另一支的节点直接是函数节点
                                        # 那么就进入循环递归的往下走另一支的节点进行判断
                                        # 如果另一支的节点直接是abs和neg这些不影响div抵消判断的函数节点
                                        while isinstance(program[temp_index], _Function) and \
                                                program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                            # 就继续往下走，直到不是abs和neg的函数节点，或者直接走到变量节点
                                            temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                        # 跳出上面的while循环之后，要么此时temp_index还是非abs和neg的函数节点
                                        # 要么就是已经抵达变量节点
                                        # 由于此时我们进入的分支是突变成变量或者常数节点（每个维度取值不一致，所以相同概率很小）
                                        # 所以如果跳出while之后是非abs和neg的函数节点，那么此时node允许生成变量节点
                                        if isinstance(program[temp_index], tuple):
                                            prohibit.append(program[temp_index])
                                    complete = True
                                    break
                            # 说明div和当前突变节点中间，还有非abs和neg的函数节点
                            if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                                # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                                while isinstance(program[temp_index], _Function) and \
                                        program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                    # 就继续往下走，直到不是abs和neg的函数节点，或者直接走到变量节点
                                    temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                # 跳过了中间的abs和neg之后，如果是一个其他的函数节点
                                # 此时temp_index和- name_index分别代表左右子树都跳过了abs和neg的下一个父亲节点
                                # 如果左右节点相同
                                if isinstance(program[temp_index], _Function) and \
                                        program[temp_index].name == name_list[- name_index]:
                                    # 相同函数名匹配后右边要重新跳过abs和neg（不能嵌套，但是会隔层出现）
                                    name_index += 1
                                    # 如果中间还有其他函数节点，那么就进入这个if判断
                                    if name_index <= len(name_list):
                                        while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                            name_index += 1
                                            if name_index > len(name_list):  # 等价于右边name_list为空
                                                # complete = True
                                                break
                                    child_index_stack = [0]  # 记录还未探索的子节点索引
                                    # 这里应该要深度优先遍历完整个子树，但碰到一条符合的之后就停止，这样足矣避免完全抵消
                                    while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                        # if name_index + 1 > len(name_list):
                                        # 如果此时name_index已经遍历到当前突变的节点
                                        if name_index > len(name_list) and complete:  # and complete是为了让temp_index跳过abs和neg
                                            children = []
                                            for c in program[temp_index].child_distance_list:  # 遍历子节点
                                                if isinstance(program[temp_index + c], _Function):
                                                    children.append(program[temp_index + c].name)
                                                elif isinstance(program[temp_index + c], tuple):
                                                    children.append(program[temp_index + c])
                                                elif program[temp_index].name != 'pow' or \
                                                        c != program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                    no_cancel = True
                                            children2 = []
                                            current = 0
                                            for c_i, c in enumerate(program[parent_index].child_distance_list):  # 遍历子节点
                                                if parent_index + c == node:  # 跳过当前点
                                                    current = c_i
                                                    continue
                                                if isinstance(program[parent_index + c], _Function):
                                                    children2.append(program[parent_index + c].name)
                                                elif isinstance(program[parent_index + c], tuple):
                                                    children2.append(program[parent_index + c])
                                                elif program[parent_index].name != 'pow' or \
                                                        c != program[parent_index].child_distance_list[-1]:  # 不是pow的指数向量
                                                    no_cancel = True
                                            if no_cancel:
                                                break
                                            if not len(children2):
                                                if isinstance(children[0], tuple):
                                                    prohibit.append(children[0])
                                                break
                                            if program[temp_index].name == program[parent_index].name:
                                                # 子节点组合匹配
                                                if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                                'sum', 'prod', 'mean']:
                                                    # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                                    for c_i, c in enumerate(children2):
                                                        # if c_i == current:
                                                        #     continue
                                                        if c not in children:  # 有一个不在
                                                            break
                                                        else:
                                                            children.remove(c)
                                                    if len(children) == 1:
                                                        if isinstance(children[0], tuple):
                                                            prohibit.append(children[0])
                                                else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                                    if program[temp_index].name == 'pow':  # pow要对指数进行检测
                                                        if node != parent_index + 1:  # 当前点是指数向量节点
                                                            if isinstance(children[0], tuple):
                                                                if children[0] == children2[0]:
                                                                    right_index = temp_index + \
                                                                                  program[
                                                                                      temp_index].child_distance_list[
                                                                                      -1]
                                                                    prohibit.append(program[right_index])
                                                        else:  # 当前点是pow的左子节点
                                                            right_index1 = temp_index + \
                                                                           program[temp_index].child_distance_list[-1]
                                                            right_index2 = parent_index + \
                                                                           program[parent_index].child_distance_list[-1]
                                                            # 指数向量相同，则避免生成相同的左变量子节点
                                                            if np.array_equal(program[right_index1],
                                                                              program[right_index2]):
                                                                if isinstance(program[temp_index + 1], tuple):
                                                                    prohibit.append(program[temp_index + 1])
                                                    else:  # 序列匹配中需要跳过当前点
                                                        for c_i, c in enumerate(children2):  # 序列匹配
                                                            if c_i < current:
                                                                if c != children[0]:
                                                                    break
                                                                else:
                                                                    children.pop(0)
                                                            else:
                                                                if c != children[1]:
                                                                    break
                                                                else:
                                                                    children.pop(1)
                                                        if len(children) == 1 and isinstance(children[0], tuple):
                                                            prohibit.append(children[0])
                                            else:  # 只有一边是neg或abs
                                                if isinstance(children[0], tuple):
                                                    prohibit.append(children[0])
                                            break
                                        match = False
                                        fence = child_index_stack.pop()  # 获取最新的
                                        temp_children = program[temp_index].child_distance_list
                                        # 检查另一支的孩子们，和name_index是否一致
                                        for j, c in enumerate(temp_children):
                                            if j < fence:
                                                continue
                                            # 如果当前另一支的跳过了abs和neg的节点的孩子也是函数节点
                                            if isinstance(program[temp_index + c], _Function):
                                                # 如果这个孩子是abs或者neg，那么就记录这个孩子节点的孩子节点（跳过，找他的下一层）
                                                if program[temp_index + c].name in ['abs', 'neg']:
                                                    match = True
                                                    child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                    child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                    temp_index += c
                                                    break
                                                # 同理，如果另一支节点的孩子节点和突变支的函数节点一致
                                                elif name_index <= len(name_list) and \
                                                        program[temp_index + c].name == name_list[- name_index]:
                                                    match = True
                                                    child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                                    child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                                    temp_index += c
                                                    name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                                    # 函数名匹配成功后右边重新跳过abs和neg
                                                    if name_index <= len(name_list):
                                                        while name_list[- name_index] in ['abs', 'neg']:
                                                            name_index += 1
                                                            if name_index > len(name_list):  # 等价于右边name_list为空
                                                                # 且左节点的子节点中没有abs和neg，则complete=True
                                                                break
                                                    break
                                        # 在上面的操作是对temp_index的孩子们，查看是否有和name_index一致的情况出现 match
                                        # 如果有的话，就会把name_index和对应一样的孩子节点，同时往下移动一层，并且跳过abs和neg的节点
                                        if not match:  # 这个当前点的子节点没有匹配的
                                            # 如果name_index已经遍历到突变节点了
                                            if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                                complete = True
                                            # 否则对于当前的另一支点和突变支的节点，需要回溯到其父节点，寻找其父节点的另一支情况？
                                            else:  # 否则还需要返回至当前点的父节点
                                                # 当前temp_index是abs或neg时右边并不需要变动
                                                if program[temp_index].name not in ['neg', 'abs']:
                                                    name_index -= 1  # 搜索上一个名字
                                                temp_index += program[temp_index].parent_distance
                                                if name_index < 0:
                                                    print("div(name_index < 0):")
                                                    self.printout(program)
                                                    print(name_list)
                                                    print(name_index)
                                                    print()
                                                    raise ValueError('name_index should be positive.')
                                                # 下面就是如果temp_index此时已经回宿到父亲节点，如果父亲节点是abs或者neg，那么就继续回溯
                                                while program[temp_index].name in ['neg', 'abs'] and \
                                                        program[temp_index].parent_distance != 0:
                                                    if len(child_index_stack):
                                                        child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                                    else:  # 超出界限说明已经不满足抵消了
                                                        break
                                                    temp_index += program[temp_index].parent_distance
                                                    # name_index -= 1  # 搜索上一个名字
                                                # 同理，name_index也继续回溯
                                                while name_list[- name_index] in ['neg', 'abs'] and name_index > 1:
                                                    name_index -= 1  # 跳过abs和neg
                                            # if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                            #     complete = True
                                            # else:  # 否则还需要返回至当前点的父节点
                                            #     temp_index += program[temp_index].parent_distance
                                            #     name_index -= 1  # 搜索上一个名字
                                            #     while program[temp_index].name in ['neg', 'abs'] and \
                                            #             program[temp_index].parent_distance != 0:
                                            #         if len(child_index_stack):
                                            #             child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                            #         else:  # 超出界限说明已经不满足抵消了
                                            #             break
                                            #         temp_index += program[temp_index].parent_distance
                                            #         name_index -= 1  # 搜索上一个名字
                        else:  # name_list为空
                            # 这种情况就是突变节点的父节点就是div
                            if node == parent_index + 1:  # 当前点是父节点的左子节点
                                another = parent_index + program[parent_index].child_distance_list[-1]
                            else:
                                another = parent_index + 1
                            if isinstance(program[another], tuple):
                                prohibit.append(program[another])
                            else:
                                # 跳过abs和neg
                                while isinstance(program[another], _Function) and \
                                        program[another].name in ['abs', 'neg']:
                                    another += 1
                                # 如果跳过之后是变量，那么和上面判断一样，如果不是变量，那就无所谓
                                if isinstance(program[another], tuple):
                                    prohibit.append(program[another])
                    # 父节点不是div，那就是sub
                    # 而且name_list不为空，代表中间会有其他的函数节点
                    elif len(name_list):  # 与sub之间存在中间函数节点
                        if isinstance(program[temp_index], _Function) and program[temp_index].name == name_list[-1]:
                            child_index_stack = [0]  # 记录还未探索的子节点索引
                            name_index = 1
                            while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                                match = False
                                # 找到了父节点均相同的子支
                                # 这里的判断意思是中间只有一个函数节点卡在sub和突变节点中间
                                if name_index + 1 > len(name_list):
                                    # 记录另外一支的子节点
                                    children = []
                                    for c in program[temp_index].child_distance_list:  # 遍历子节点
                                        if isinstance(program[temp_index + c], _Function):
                                            children.append(program[temp_index + c].name)
                                        elif isinstance(program[temp_index + c], tuple):
                                            children.append(program[temp_index + c])
                                        # 如果当前temp_index不是pow函数
                                        # 或者这个孩子不是pow函数的指数向量位置
                                        elif program[temp_index].name != 'pow' or \
                                                c != program[temp_index].child_distance_list[-1]:  # 不是pow的指数向量
                                            no_cancel = True
                                    children2 = []
                                    current = 0
                                    for c_i, c in enumerate(program[parent_index].child_distance_list):  # 遍历子节点
                                        if parent_index + c == node:  # 跳过当前点
                                            current = c_i
                                            continue
                                        if isinstance(program[parent_index + c], _Function):
                                            children2.append(program[parent_index + c].name)
                                        elif isinstance(program[parent_index + c], tuple):
                                            children2.append(program[parent_index + c])
                                        elif program[parent_index].name != 'pow' or \
                                                c != program[parent_index].child_distance_list[-1]:  # 不是pow的指数向量
                                            no_cancel = True
                                    if no_cancel:
                                        break
                                    if not len(children2):
                                        if isinstance(children[0], tuple):
                                            prohibit.append(children[0])
                                        break
                                    # 子节点组合匹配
                                    if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                    'sum', 'prod', 'mean']:
                                        # 只对非常数节点做检测，若子节点组合相同，则cancel=True
                                        for c_i, c in enumerate(children2):
                                            # if c_i == current:
                                            #     continue
                                            if c not in children:  # 有一个不在
                                                break
                                            else:
                                                children.remove(c)
                                        if len(children) == 1:
                                            if isinstance(children[0], tuple):
                                                prohibit.append(children[0])
                                    else:  # 其他函数则进行序列匹配，pow的指数不会被children和children2记录，要单独检测
                                        if program[temp_index].name == 'pow':  # pow要对指数进行检测
                                            if node != parent_index + 1:  # 当前点是指数向量节点
                                                if isinstance(children[0], tuple):
                                                    if children[0] == children2[0]:
                                                        right_index = temp_index + \
                                                                      program[temp_index].child_distance_list[-1]
                                                        prohibit.append(program[right_index])
                                            else:  # 当前点是pow的左子节点
                                                right_index1 = temp_index + \
                                                               program[temp_index].child_distance_list[-1]
                                                right_index2 = parent_index + \
                                                               program[parent_index].child_distance_list[-1]
                                                # 指数向量相同，则避免生成相同的左变量子节点
                                                if np.array_equal(program[right_index1], program[right_index2]):
                                                    if isinstance(program[temp_index + 1], tuple):
                                                        prohibit.append(program[temp_index + 1])
                                        else:  # 序列匹配中需要跳过当前点
                                            for c_i, c in enumerate(children2):  # 序列匹配
                                                if c_i < current:
                                                    if c != children[0]:
                                                        break
                                                    else:
                                                        children.pop(0)
                                                else:
                                                    if c != children[1]:
                                                        break
                                                    else:
                                                        children.pop(1)
                                            if len(children) == 1 and isinstance(children[0], tuple):
                                                prohibit.append(children[0])
                                    break
                                temp_name = name_list[- name_index - 1]
                                fence = child_index_stack.pop()  # 获取最新的
                                temp_children = program[temp_index].child_distance_list
                                for j, c in enumerate(temp_children):
                                    if j < fence:
                                        continue
                                    if isinstance(program[temp_index + c], _Function) and \
                                            program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                        child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                        temp_index = temp_index + c
                                        child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                        name_index += 1  # 搜索下一个名字
                                        match = True
                                        break
                                if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                                    temp_index = temp_index + program[temp_index].parent_distance  # 确保初始值正确
                                    name_index -= 1  # 搜索上一个名字
                    else:  # 与sub之间没有中间函数节点
                        if node == parent_index + 1:  # 当前点是父节点的左子节点
                            right = parent_index + program[parent_index].child_distance_list[-1]
                            if isinstance(program[right], tuple):
                                prohibit.append(program[right])
                        elif isinstance(program[parent_index + 1], tuple):  # 是右子节点，则避免生成与左子节点相同的点
                            prohibit.append(program[parent_index + 1])
                if parent_name in ['max', 'min']:
                    if node == parent_index + 1:  # 左子节点
                        another = parent_index + program[parent_index].child_distance_list[-1]  # 右子节点
                    else:  # 右子节点
                        another = parent_index + 1
                    if isinstance(program[another], tuple) and program[another] not in prohibit:
                        prohibit.append(program[another])

                if parent_name == 'add':  # 限制最大维度的X+X
                    if node == parent_index + 1:  # 左子节点
                        another = parent_index + program[parent_index].child_distance_list[-1]  # 右子节点
                    else:  # 右子节点
                        another = parent_index + 1
                    if isinstance(program[another], tuple) and program[another] not in prohibit:
                        if self.calculate_dimension(program[another]) == self.n_features:
                            prohibit.append(program[another])
                # 不是根节点，parent_index才有意义
                if node != 0 and parent_name == 'pow' and node != parent_index + 1:  # 若当前节点是pow的指数向量
                    terminal = self.generate_a_terminal(random_state, existed_dimension,
                                                        const_int=True, prohibit=prohibit)
                # 父节点是arity为1的函数节点或pow函数节点，或者父节点constant_num已经为1且要突变的不是常数节点，则子节点不能突变为常数
                elif program[parent_index].arity == 1 or parent_name == 'pow' or \
                        program[parent_index].constant_num >= 1 and not isinstance(program[node], list):
                    no_mutate = False
                    if len(prohibit) and existed_dimension == self.n_features:
                        for item in prohibit:
                            # if isinstance(item, tuple) and (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                            if isinstance(item, tuple) and self.calculate_dimension(item) == self.n_features:
                                no_mutate = True  # 既不能突变为向量切片也不能突变为常数节点，则该点不突变
                                break
                    if no_mutate:
                        continue
                    terminal = self.generate_a_terminal(random_state, existed_dimension, vary=True, prohibit=prohibit)
                else:  # arity>1的函数节点，可以生成常数节点，也可以生成向量节点，但现在向量节点有生成限制，要判断能不能生成向量节点
                    # self.printout(program)
                    # print(node)
                    terminal = self.generate_a_terminal(random_state, existed_dimension, prohibit=prohibit)
                    # # 不符合向量节点的生成，则只能生成常数节点
                    # if existed_dimension != self.n_features and existed_dimension != self.n_features - 1:
                    #     terminal = self.generate_a_terminal(random_state, existed_dimension, const=True, prohibit=prohibit)
                    # else:
                    #     terminal = self.generate_a_terminal(random_state, existed_dimension, prohibit=prohibit)
                program[node] = terminal
                if node != 0:  # 当前点不是根节点，则需要维护当前点的父节点的constant_num属性
                    temp_constant_num = 0
                    for i in program[parent_index].child_distance_list:  # 重新对start父节点的常数子节点进行计数
                        if isinstance(program[parent_index + i], list):
                            temp_constant_num += 1
                    program[parent_index].constant_num = temp_constant_num
        # y_pred = self.execute_test(program, X_train, random_state)
        # if np.max(y_pred) - np.min(y_pred) <= 1e-8 and check_constant_function:
        #     print("-----常数函数(point)-----")
        #     self.printout(self.program)
        #     self.printout(program)
        #     self.print_formula(program, show_operand=True)
        return program, list(mutate)

    # 辅助函数
    def get_output_dimensions(self, program=None):
        if program is None:
            program = self.program
        output_dimensions = np.full(self.n_features, 0)
        for index, node in enumerate(program):  # 遍历该公式，找到所有不同输出维度的节点
            output_dimensions[self.calculate_dimension(node) - 1] += 1
        result = []
        for index, item in enumerate(output_dimensions):
            if item != 0:  # 有这个输出维度的节点
                result.append(index + 1)  # 记录该输出维度
        return result

    def get_depth(self, index, program=None):  # 给定的program和index，找到index对应节点(函数，常数或变量)的深度
        if program is None:
            program = self.program
        if index == 0:
            return 0
        if len(program) < index:
            raise ValueError("Get_depth: The length of program is smaller than the value of the index.")
        if isinstance(program[0], _Function):  # 初始深度应为program[0]的深度
            current_depth = program[0].depth
        else:
            current_depth = 0
        terminal_stack = []
        for i, node in enumerate(program):
            if i == index:  # 索引相同，找到了要找的点，返回当前深度
                return current_depth
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
                current_depth += 1
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    current_depth -= 1
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时仍未找到，返回None
                        return None  # We should never get here
                    terminal_stack[-1] -= 1
        return current_depth  # 没找到该点，说明该点是最后一个点，那返回最新的current_depth即可

    def set_depth(self, init_depth, program=None):  # 给定根节点深度和program，设置program这两个属性
        if program is None:
            program = self.program
        if len(program) == 1:
            return deepcopy(program)
        current_depth = init_depth
        terminal_stack = []
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        for index, node in enumerate(program):
            if isinstance(node, _Function):
                new_program[index].depth = current_depth  # 不能返回或修改原数组，应该返回新的数组，避免修改共用数组导致错误
                terminal_stack.append(node.arity)
                current_depth += 1
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    current_depth -= 1
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return new_program
                    terminal_stack[-1] -= 1

    def set_remaining(self, remaining, program=None):  # 给定根节点remaining和program，设置program中所有点的remaining值
        if program is None:
            program = self.program
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        if isinstance(program[0], _Function):  # 函数节点
            new_program[0].remaining = remaining
        else:
            return new_program
        terminal_stack = []
        for index, node in enumerate(program):
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
                if index == 0:  # 第一个点已经设置了
                    continue
                parent_index = self.find_parent(index, new_program)
                parent_remaining = new_program[parent_index].remaining
                current_remaining = self.update_remaining(parent_remaining, node)
                new_program[index].remaining = current_remaining
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return new_program
                    terminal_stack[-1] -= 1
        return None  # We should never get here

    def set_total(self, index, program=None, subtract=False):  # 给定当前点索引以及program，设置当前点以及其所有ancestors的total值
        if program is None:
            program = self.program
        new_program = deepcopy(program)  # 深复制，避免修改共用数组导致错误
        if isinstance(new_program[index], _Function):  # 当前节点是函数节点才会需要修改ancestors的total值
            parent_index = self.find_parent(index, new_program)  # 找父节点，根节点会返回自己，然后会修改自己的total值，而其他节点不会
            if parent_index != index:  # 其他节点也修改自身的total值，与根节点保持一致
                new_program[index].total = self.update_total(new_program[index].total, new_program[index], subtract)
            # 给定节点total和当前函数节点，更新节点total值
            new_program[parent_index].total = self.update_total(new_program[parent_index].total,
                                                                new_program[index], subtract)  # 更新父节点total属性
            while parent_index != 0:  # 只有parent_index到0才会停止
                parent_index = self.find_parent(parent_index, new_program)  # 找父节点的父节点
                new_program[parent_index].total = self.update_total(new_program[parent_index].total,
                                                                    new_program[index], subtract)  # 更新父节点total属性
        return new_program

    # const_int指定生成pow函数需要的整数向量，vary指定生成变量向量，const_range长度不为0则需要根据const_range来生成常数。
    def generate_a_terminal(self, random_state, output_dimension, const_int=False, vary=False,
                            const_range=(), const=False, prohibit=()):
        if len(prohibit) and output_dimension == self.n_features:
            for item in prohibit:
                # if isinstance(item, tuple) and (item[1] - item[0] - 1) / item[2] + 1 == self.n_features:
                if isinstance(item, tuple) and self.calculate_dimension(item) == self.n_features:
                    if not vary:
                        const = True  # 禁止了最大维度，则只能生成常数
                    else:
                        print('Cancel error.')
                    break
        if vary and const:
            raise ValueError('Parameter "vary" and "const" cannot be both True.')
        if not const_int:
            if not vary:
                # [0, self.n_features]，self.n_features表示选择常数向量
                if not const:
                    terminal = random_state.randint(self.n_features + 1)
                else:  # const为True生成常数
                    terminal = self.n_features
            else:  # 若vary为True，则不生成常数向量
                terminal = random_state.randint(self.n_features)
            if terminal == self.n_features:  # 常数向量，但需要外面套一层[]，使其类型变为list，避免execute函数中类型混淆
                terminal = []
                if len(const_range):  # 有大小要求则按大小要求来
                    min_exponent = int(np.floor(np.log10(max(const_range[0],0.1))))
                    max_exponent = int(np.floor(np.log10(const_range[1])))
                        
                    for i in range(output_dimension):
                        while True:
                            mantissa = random_state.randint(1, 12)  # 10表示π，11表示e
                            exponent = random_state.randint(min_exponent, max_exponent + 1)  # 数量级范围是[min_exponent, max_exponent]
                            if mantissa == 10:
                                mantissa = np.pi
                            elif mantissa == 11:
                                mantissa = np.e
                            if random_state.randint(0, 2) == 0:
                                result = mantissa * 10 ** exponent
                            else:
                                result = - mantissa * 10 ** exponent
                            # 确保生成的数字在新的范围内
                            if const_range[0] <= abs(result) <= const_range[1]:
                                terminal.append(result)
                                break
                else:  # 随机生成时采用科学记数法
                    for i in range(output_dimension):
                        mantissa = random_state.randint(1, 12)  # 10表示π，11表示e
                        exponent = random_state.randint(-1, 4)  # 数量级范围是[-1, 3]
                        if mantissa == 10:
                            mantissa = np.pi
                        elif mantissa == 11:
                            mantissa = np.e
                        if random_state.randint(0, 2) == 0:
                            result = mantissa * 10 ** exponent
                        else:
                            result = - mantissa * 10 ** exponent
                        terminal.append(result)
                terminal = [np.array(terminal, dtype=np.float64)]  # float64数据类型，统一类型
            else:  # 随机生成向量切片，已知维度->切片
                # 随机生成合法步长，[1, ]
                same = True
                start, end, step = 0, 0, 0
                counter = 0
                while same:  # 生成与prohibit相同的变量节点时重新生成
                    same = False
                    counter += 1
                    if self.n_features > 1:
                        step = 1  # 切片步长只能是1，起点从0或1中选，终点固定为最后一个维度
                        # 起点是0或1，当self.n_features>2时变量节点只能出现在聚合运算符的子树中
                        # start = random_state.randint(2)
                        if output_dimension == self.n_features:
                            start = 0
                            # end = self.n_features  # 索引最大值是self.n_features - 1
                            end = 0  # end表示从右开始数多少个是终点，0表示末尾
                        elif output_dimension == self.n_features - 1:
                            start = random_state.randint(2)
                            # end = self.n_features - 1 + start
                            end = (start + 1) % 2  # start=0,end=1;start=1,end=0
                        else:
                            print(output_dimension)
                            print(self.n_features)
                            raise ValueError('Parameter "output_dimension" should equal to self.n_features '
                                             'or (self.n_features - 1) numerically.')
                            # if output_dimension > 1:  # output_dimension的要求和切片的要求可能是冲突的
                        #     # step = random_state.randint(math.floor(self.n_features / output_dimension)) + 1
                        #     step = 1  # 切片步长只能是1，起点从0或1中选，终点固定为最后一个维度
                        #     # start = random_state.randint(self.n_features - (output_dimension - 1) * step)  # 随机选择切片起点
                        #     start = random_state.randint(2)  # 起点是0或1，当self.n_features>2时变量节点只能出现在聚合运算符的子树中
                        #     end = self.n_features  # 索引最大值是self.n_features - 1
                        #     # start = end - output_dimension
                        #     # end = start + (output_dimension - 1) * step + 1  # 计算切片终点，由于左闭右开，需要+1
                        # else:  # output_dimension == 1
                        #     step = 1
                        #     # start = random_state.randint(self.n_features)  # 随机选择切片起点
                        #     end = self.n_features  # 计算切片终点，由于左闭右开，需要+1
                        #     start = end - 1
                    else:
                        step = 1
                        start = 0
                        # end = 1
                        end = 0
                    if len(prohibit):
                        for item in prohibit:
                            if (start, end, step) == item:  # 如果有相同的就重新生成
                                same = True
                                break
                    if counter >= 10:
                        # print(f"Endless loop error from variables.{start, end, step}")
                        break
                terminal = (start, end, step)  # 记录该切片
        else:  # 生成pow需要的指数整数向量
            terminal = random_state.randint(2, 4, output_dimension)
            neg = random_state.uniform(size=output_dimension)
            terminal = [np.array(np.where(neg > 0.5, terminal, -terminal))]  # 一半的概率指数变为相反数，范围{-4,-3,-2,2,3,4}
            counter = 0
            if len(prohibit):
                while np.array_equal(terminal[0], prohibit[0][0]):
                    counter += 1
                    terminal = random_state.randint(2, 4, output_dimension)
                    neg = random_state.uniform(size=output_dimension)
                    terminal = [np.array(np.where(neg > 0.5, terminal, -terminal))]  # 一半的概率指数变为相反数，范围{-4,-3,-2,2,3,4}
                    if counter >= 10:
                        print("Endless loop error from pow.")
                        break
        return terminal

    # 给定program和当前点索引，返回当前点的父节点的索引，调用了get_depth函数
    def find_parent(self, current_index, program=None):
        if current_index == 0:
            return 0
        if program is None:
            program = self.program
        current_depth = self.get_depth(current_index, program)
        if current_depth is None:
            print(current_index, end=" ")
        assert current_depth is not None
        for index in range(1, current_index + 1):  # [1, current_index]，寻找父节点函数
            if isinstance(program[current_index - index], _Function):  # 函数
                if program[current_index - index].depth == current_depth - 1:  # 若该函数节点为当前节点的父节点
                    parent_index = current_index - index  # 记录父节点函数索引
                    return parent_index

    # 给定父节点remaining，父节点名字，当前点total值和function_set，给出合法的function_set
    def clip_function_set(self, remaining, function_set=None, no_pow=False, parent_name=None, total=None):
        if function_set is None:
            function_set = self.function_set  # _Function对象列表
        if total is None:   
            total = deepcopy(default_total)
        prohibit = []
        if remaining[0] <= 0 or total[0] >= default_remaining[0]:  # aggregate次数为0
            for name in aggregate:
                prohibit.append(name)
        # 或者改成 < 2 
        elif remaining[0] < 2 or total[0] > 2:  # aggregate次数不足以支持prod和mean和sum
            prohibit.append('prod')
            prohibit.append('mean')
            prohibit.append('sum')
        # elif remaining[0] < 3 or total[0] > 0:  # aggregate次数不足以支持prod和mean
        #     prohibit.append('prod')
        #     prohibit.append('mean')
        if remaining[1] <= 0 or total[1] >= default_remaining[1] or no_pow:  # pow次数为0或指定没有pow函数
            prohibit.append('pow')
        if remaining[2] <= 0 or total[2] >= default_remaining[2]:  # 基本初等函数次数为0
            for name in elementary_functions:
                prohibit.append(name)
        if remaining[3] <= 0 or total[3] >= default_remaining[3]:  # 剩余exp次数为0
            prohibit.append('exp')
        # if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
        #     if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt','mean']:
        #         prohibit.append(parent_name)  # 防止这些函数连续嵌套
        #     elif parent_name == 'exp':  # 防止exp和log连续嵌套
        #         prohibit.append('log')
        #     elif parent_name == 'log':
        #         prohibit.append('exp')
        # 2024.12.2 可以根据父节点来限制可生成的function_set,类似于crossover的那种筛选
        if parent_name is not None:  # 避免相同函数或互逆函数连续嵌套
            # 此处其实应该把pow也加进去？不然会导致变化太大了？但是如果直接在remain设置了应该也不会出现嵌套
            if parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt','mean']:
                prohibit.append(parent_name)  # 防止这些函数连续嵌套
            elif parent_name == 'exp':  # 防止exp和log连续嵌套
                prohibit.append('log')
            elif parent_name == 'log':
                prohibit.append('exp')
                prohibit.append('neg')
            # 下面是避免冗余操作
            elif parent_name == 'sqrt' :
                if 'abs' not in prohibit:
                    prohibit.append('abs')
                if 'neg' not in prohibit:
                    prohibit.append('neg')
            elif parent_name == 'abs' and 'neg' not in prohibit:
                prohibit.append('neg')
        
        
        if len(prohibit) == 0:
            return function_set
        new_function_set = []
        for item in function_set:
            if item.name not in prohibit:  # 不在禁止范围内
                new_function_set.append(item)
        return new_function_set  # 返回约束范围后的函数_Function对象集

    def calculate_dimension(self, node):
        if isinstance(node, _Function):  # 函数
            return node.output_dimension
        elif isinstance(node, tuple):  # 变量向量
            return math.ceil((self.n_features - node[1] - node[0]) / node[2])
        elif isinstance(node, list):  # 常数向量
            return len(node[0])
        return None  # We should never get here

    @staticmethod  # 给定当前点父节点的remaining，以及当前点的函数节点，返回更新后当前点的remaining值
    def update_remaining(remaining, function):
        name = function.name
        new_remaining = deepcopy(remaining)
        if name in [ 'min', 'max']:
            new_remaining[0] -= 1
        elif name in ['mean','sum', 'prod']:
            new_remaining[0] -= 2
        elif name == 'pow':
            new_remaining[1] -= 1
        elif name in ['sin', 'cos', 'tan', 'log','tanh']:
            new_remaining[2] -= 1  # 基本初等函数次数减1
        elif name == 'exp':
            new_remaining[3] -= 1
        return new_remaining

    @staticmethod  # 给定父节点total值，以及当前函数节点，更新父节点的total值
    def update_total(total, function, subtract=False):
        name = function.name
        new_total = deepcopy(total)
        if subtract:
            if name in [ 'min', 'max']:
                new_total[0] -= 1
                # 修改权重
            elif name in ['mean','sum', 'prod']:
                new_total[0] -= 2
            elif name == 'pow':
                new_total[1] -= 1
            elif name in ['sin', 'cos', 'tan', 'log','tanh']:
                new_total[2] -= 1  # 基本初等函数次数减1
            elif name == 'exp':
                new_total[3] -= 1
        else:
            if name in [ 'min', 'max']:
                new_total[0] += 1
                # 修改权重
            elif name in ['mean', 'sum','prod']:
                new_total[0] += 2
            elif name == 'pow':
                new_total[1] += 1
            elif name in ['sin', 'cos', 'tan', 'log','tanh']:
                new_total[2] += 1  # 基本初等函数次数加1
            elif name == 'exp':
                new_total[3] += 1
        return new_total

    def check_total(self, program=None):  # 给定program，检测根节点total值是否正确
        if program is None:
            program = self.program
        if len(program) == 1:
            return True
        total = [0, 0, 0, 0]
        terminal_stack = []
        for index, node in enumerate(program):  # 遍历该树来统计相应点然后与根节点total值比较
            if isinstance(node, _Function):
                total = self.update_total(total, node)  # 统计该点
                terminal_stack.append(node.arity)
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:  # terminal_stack为空时返回program
                        return total == program[0].total
                    terminal_stack[-1] -= 1

    # left_child_index和right_child_index是sub或div的左右子节点在program中的索引
    @staticmethod
    def check_sub_div_cancel(program, left_child_index, right_child_index):
        cancel = False
        root = left_child_index - 1
        left_child, right_child = program[left_child_index], program[right_child_index]
        if isinstance(left_child, tuple) and left_child == right_child:  # 变量节点相同
            cancel = True
        # 两边进行深度优先搜索
        elif isinstance(left_child, _Function) and isinstance(right_child, _Function):
            if left_child.name == right_child.name or \
                    program[root].name == 'div' and \
                    (left_child.name in ['abs', 'neg'] or right_child.name in ['abs', 'neg']):
                left_child_index_stack = [0]  # 记录左子树还未探索的子节点索引
                right_child_index_stack = [0]  # 记录右子树还未探索的子节点索引
                # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                counter = 0
                while len(left_child_index_stack) and len(right_child_index_stack):
                    match = False
                    counter += 1
                    if counter >= 50:
                        # printout(program=program, max_dim=)
                        print(left_child_index)
                        print(right_child_index)
                        print(left_child.child_distance_list)
                        print(right_child.child_distance_list)
                        print(left_child_index_stack)
                        print(right_child_index_stack)
                    if counter >= 60:
                        # 此时如果还检测不出来，那就说明很复杂，不可能抵消了
                        return False
                        # raise ValueError('Endless Loop.')
                    left_fence = left_child_index_stack.pop()  # 获取最新的
                    right_fence = right_child_index_stack.pop()  # 获取最新的
                    for l_index, l_child_distance in enumerate(left_child.child_distance_list):
                        if l_index < left_fence:
                            continue
                        l_child_index = left_child_index + l_child_distance
                        l_child = program[l_child_index]
                        for r_index, r_child_distance in enumerate(right_child.child_distance_list):
                            if r_index < right_fence:
                                continue
                            r_child_index = right_child_index + r_child_distance
                            r_child = program[r_child_index]
                            if isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
                                cancel = True
                                break
                            elif isinstance(l_child, _Function) and isinstance(r_child, _Function):
                                if program[root].name == 'div':
                                    while isinstance(l_child, _Function) and l_child.name in ['neg', 'abs']:
                                        left_child_index_stack.append(1)  # neg和abs是单节点函数
                                        l_child_index = l_child_index + 1
                                        l_child = program[l_child_index]
                                    while isinstance(r_child, _Function) and r_child.name in ['neg', 'abs']:
                                        right_child_index_stack.append(1)  # neg和abs是单节点函数
                                        r_child_index = r_child_index + 1
                                        r_child = program[r_child_index]
                                    if isinstance(l_child, _Function) and isinstance(r_child, _Function):
                                        if l_child.name == r_child.name:
                                            left_child_index = l_child_index
                                            left_child = l_child
                                            left_child_index_stack.append(l_index + 1)
                                            left_child_index_stack.append(0)

                                            right_child_index = r_child_index
                                            right_child = r_child
                                            right_child_index_stack.append(r_index + 1)
                                            right_child_index_stack.append(0)
                                            match = True
                                            break
                                    elif isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
                                        cancel = True
                                        break
                                elif l_child.name == r_child.name:
                                    left_child_index = l_child_index
                                    left_child = l_child
                                    left_child_index_stack.append(l_index + 1)
                                    left_child_index_stack.append(0)

                                    right_child_index = r_child_index
                                    right_child = r_child
                                    right_child_index_stack.append(r_index + 1)
                                    right_child_index_stack.append(0)
                                    match = True
                                    break
                        if match or cancel:
                            break
                    if cancel:  # 发现存在抵消，则停止搜索，并禁止此次突变
                        break
                    if not match:  # 没找到匹配的就回溯到父节点继续搜索
                        left_child_index += left_child.parent_distance
                        left_child = program[left_child_index]
                        right_child_index += right_child.parent_distance
                        right_child = program[right_child_index]
                        if program[root].name == 'div':  # 只有div才忽略neg和abs
                            while left_child.name in ['neg', 'abs']:
                                left_child_index += left_child.parent_distance
                                left_child = program[left_child_index]
                                if len(left_child_index_stack):
                                    left_child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                else:
                                    break
                            while right_child.name in ['neg', 'abs']:
                                right_child_index += right_child.parent_distance
                                right_child = program[right_child_index]
                                if len(right_child_index_stack):
                                    right_child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                else:
                                    break
            # if left_child.name == right_child.name:
            #     left_child_index_stack = [0]  # 记录左子树还未探索的子节点索引
            #     right_child_index_stack = [0]  # 记录右子树还未探索的子节点索引
            #     # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
            #     while len(left_child_index_stack) and len(right_child_index_stack):
            #         match = False
            #         left_fence = left_child_index_stack.pop()  # 获取最新的
            #         right_fence = right_child_index_stack.pop()  # 获取最新的
            #         for l_index, l_child_distance in enumerate(left_child.child_distance_list):
            #             if l_index < left_fence:
            #                 continue
            #             l_child_index = left_child_index + l_child_distance
            #             l_child = program[l_child_index]
            #             for r_index, r_child_distance in enumerate(right_child.child_distance_list):
            #                 if r_index < right_fence:
            #                     continue
            #                 r_child_index = right_child_index + r_child_distance
            #                 r_child = program[r_child_index]
            #                 if isinstance(l_child, tuple) and l_child == r_child:  # 变量节点相同
            #                     cancel = True
            #                     break
            #                 elif isinstance(l_child, _Function) and isinstance(r_child, _Function):
            #                     if l_child.name == r_child.name:
            #                         left_child_index = l_child_index
            #                         left_child = l_child
            #                         left_child_index_stack.append(l_index + 1)
            #                         left_child_index_stack.append(0)
            #
            #                         right_child_index = r_child_index
            #                         right_child = r_child
            #                         right_child_index_stack.append(r_index + 1)
            #                         right_child_index_stack.append(0)
            #                         match = True
            #                         break
            #             if match or cancel:
            #                 break
            #         if cancel:  # 发现存在抵消，则停止搜索，并禁止此次突变
            #             break
            #         if not match:  # 没找到匹配的就回溯到父节点继续搜索
            #             left_child_index = left_child_index + left_child.parent_distance
            #             left_child = program[left_child_index]
            #             right_child_index = right_child_index + right_child.parent_distance
            #             right_child = program[right_child_index]
        return cancel

    def get_sub_div_prohibit(self, program, current_index, index_list, name_list):
        prohibit = []
        # children记录的是当前点的直接父亲的其他孩子变量节点信息
        children = []
        no_cancel = False
        parent_index = self.find_parent(current_index, program)
        # program[parent_index].child_distance_list中包括了当前点索引，所以要用[:-1]去掉最后一个
        for c in program[parent_index].child_distance_list[:-1]:  # 这里要保证当前点是父节点的最后一个子节点

            if not isinstance(program[parent_index + c], tuple):  # children会比children2少最后一个操作数
                no_cancel = True
                break
            else:  # 只记录变量节点
                children.append(program[parent_index + c])
        
        # 第一个判断是，如果当前要生成的点的直接父亲就是div或者sub，并且还位于右子树位置
        # 第二个判断是，div或者sub不是直接父亲，并且当前点位于div和sub的右子树位置
        if len(index_list) == 1 and current_index != index_list[-1] + 1 or \
                len(index_list) > 1 and index_list[-2] != index_list[-1] + 1:
            # temp_index是div或者sub的直接左子树节点
            temp_index = index_list[-1] + 1
            # 如果是判断div
            if program[index_list[-1]].name == 'div':
                # 如果中间有其他父节点
                if len(name_list):
                    complete = False
                    name_index = 1
                    # 从上至下检测中间的父节点
                    while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                        name_index += 1
                        # 如果跳过abs和neg之后，已经遍历到当前要生成的点的位置，那么就要开始检测左子树了
                        # abs和neg没有连续嵌套，所以这段代码只对name_list为1，只有一个中间父节点，并且是abs和neg的生效
                        if name_index > len(name_list):  # 等价于右边name_list为空
                            # 如果左子树是变量，那么当前点避免生成变量
                            if isinstance(program[temp_index], tuple):
                                prohibit.append(program[temp_index])
                            # 如果左子树是函数节点
                            elif isinstance(program[temp_index], _Function):
                                # 同理，跳过左子树的abs和neg
                                while isinstance(program[temp_index], _Function) and \
                                        program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                                    temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                                # 跳过了abs和neg之后的左子树节点，进行判断类型
                                # 如果是变量节点，那就避免生成导致抵消，其他情况则不考虑
                                if isinstance(program[temp_index], tuple):
                                    prohibit.append(program[temp_index])
                            complete = True
                            break
                    # 如果中间父节点不止一层，并且跳过了abs和neg之后，name_index还没抵达要生成节点位置
                    if not complete:  # complete为False说明右边还有待判断的函数，且不是abs或neg
                        # 当前temp_index是div的左子节点，所以可以通过这种方式跳过abs和neg
                        while isinstance(program[temp_index], _Function) and \
                                program[temp_index].name in ['abs', 'neg']:  # 左边跳过abs和neg
                            temp_index += 1  # abs和neg是单节点函数，左边是完整子树，不会越界
                        
                        # 跳过了左子树的abs和neg，和当前右子树的中间父亲对比，如果右子树是函数节点并且和中间父亲一样
                        if isinstance(program[temp_index], _Function) and \
                                program[temp_index].name == name_list[- name_index]:
                            # 相同函数名匹配后右边要重新跳过abs和neg
                            name_index += 1
                            # 继续看下一个中间父亲，并且得跳过abs和neg
                            if name_index <= len(name_list):
                                while name_list[- name_index] in ['abs', 'neg']:  # 右边跳过abs和neg
                                    name_index += 1
                                    if name_index > len(name_list):  # 等价于右边name_list为空
                                        # complete = True
                                        break
                            child_index_stack = [0]  # 记录还未探索的子节点索引
                            while len(child_index_stack) or complete:  # 若complete=True，则进行最后一次循环
                                # 如果中间节点还没遍历到当前节点
                                if name_index > len(name_list) and complete:
                                    # 存放会出现抵消的变量孩子节点
                                    children2 = []
                                    # 遍历右子树的所有子节点，此时右子树已经和左子树中间节点一样（但是中间节点已经跳到下一个节点了）
                                    for c in program[temp_index].child_distance_list:  # 遍历子节点
                                        # children2要加上对pow的检测
                                        # 如果这个孩子不是变量节点
                                        if not isinstance(program[temp_index + c], tuple):
                                            # 如果当前右子树节点是pow函数
                                            if isinstance(program[temp_index], _Function) and \
                                                    program[temp_index].name == 'pow':  # 父节点是pow函数
                                                # pow函数的最后一个操作数，即指数节点
                                                if c == program[temp_index].child_distance_list[-1]:
                                                    # 把这个pow的指数位置的常数向量加到children2中
                                                    # 防止生成的是一样的指数常数节点？
                                                    children2.append(program[temp_index + c])
                                                else:
                                                    no_cancel = True
                                                    break
                                            # 如果右子树节点不是变量和不是pow，是其他的函数或者常数，可以直接no_cancel
                                            else:
                                                no_cancel = True
                                                break
                                        # 如果这个孩子是变量节点，那么就需要存放
                                        else:
                                            children2.append(program[temp_index + c])
                                    # 如果检查完右子树的所有孩子
                                    if no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                                        break
                                    if not len(children):  # children为空，说明当前函数arity为1
                                        # if program[temp_index].name == program[parent_index].name:
                                        # 直接避免生成右子树的其中一个孩子，就可避免完全抵消？
                                        prohibit.append(children2[0])  # 可以避免完全抵消
                                        break
                                    if program[temp_index].name == program[parent_index].name:
                                        # 子节点组合匹配，子节点都是变量节点才需要检测抵消
                                        if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                        'sum', 'prod', 'mean']:
                                            for c in children:
                                                if c in children2:  # 在children2中，则将children2中相同的节点去除
                                                    children2.remove(c)
                                                if c == children[-1] and len(children2) == 1:
                                                    prohibit.append(children2[0])
                                        else:  # 其他函数则进行序列匹配
                                            for c_i, c in enumerate(children):
                                                if c != children2[c_i]:  # # 如果有一个不相同，则不会抵消
                                                    break
                                                if c == children[-1]:  # 前面都相同，则禁止生成children2的最后一个
                                                    prohibit.append(children2[-1])
                                    else:  # 左右两边函数名不相同，则右边是abs或neg中的一个
                                        prohibit.append(children2[0])
                                    break
                                match = False
                                fence = child_index_stack.pop()  # 获取最新的
                                temp_children = program[temp_index].child_distance_list
                                for j, c in enumerate(temp_children):
                                    if j < fence:
                                        continue
                                    if isinstance(program[temp_index + c], _Function):
                                        # 忽略abs和neg时右边不变，所以match=False时也要判断当前temp_index是不是abs或neg
                                        if program[temp_index + c].name in ['abs', 'neg']:
                                            match = True
                                            child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                            child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                            temp_index += c
                                            break
                                        elif name_index <= len(name_list) and \
                                                program[temp_index + c].name == name_list[- name_index]:
                                            match = True
                                            child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                            child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                            temp_index += c
                                            name_index += 1  # 搜索下一个不是abs和neg的函数名字
                                            # 函数名匹配成功后右边重新跳过abs和neg
                                            if name_index <= len(name_list):
                                                while name_list[- name_index] in ['abs', 'neg']:
                                                    name_index += 1
                                                    if name_index > len(name_list):  # 等价于右边name_list为空
                                                        # 且左节点的子节点中没有abs和neg，则complete=True
                                                        break
                                            break
                                if not match:  # 这个当前点的子节点没有匹配的
                                    if name_index > len(name_list):  # 这表明没有abs或neg函数，则可以停止搜索
                                        complete = True
                                    else:  # 否则还需要返回至当前点的父节点
                                        # 当前temp_index是abs或neg时右边并不需要变动
                                        if program[temp_index].name not in ['neg', 'abs']:
                                            name_index -= 1  # 搜索上一个名字
                                        temp_index += program[temp_index].parent_distance
                                        if name_index <= 0:
                                            print("div(name_index < 0):")
                                            self.printout(program)
                                            print(name_list)
                                            print(name_index)
                                            print()
                                            raise ValueError('name_index should be positive.')
                                        while program[temp_index].name in ['neg', 'abs'] and \
                                                program[temp_index].parent_distance != 0:
                                            if len(child_index_stack):
                                                child_index_stack.pop()  # 去掉neg和abs记录的下一探索点
                                            else:  # neg或abs由循环外的步骤跳过，在此处重新回到neg再pop会导致越界错误
                                                break
                                            temp_index += program[temp_index].parent_distance
                                            # name_index -= 1  # 搜索上一个名字
                                        while name_list[- name_index] in ['neg', 'abs'] and name_index > 1:
                                            name_index -= 1  # 跳过abs和neg
                else:  # 右边name_list为空
                    if isinstance(program[temp_index], tuple):
                        prohibit.append(program[temp_index])
                    else:
                        while isinstance(program[temp_index], _Function) and \
                                program[temp_index].name in ['abs', 'neg']:
                            temp_index += 1
                        if isinstance(program[temp_index], tuple):
                            prohibit.append(program[temp_index])
            elif len(name_list):  # 与sub之间存在中间函数节点
                # 如果当前右子树节点和左子树节点一致
                # 那么就需要继续往下搜一层
                if isinstance(program[temp_index], _Function) and program[temp_index].name == name_list[-1]:
                    child_index_stack = [0]  # 记录还未探索的子节点索引
                    name_index = 1
                    while len(child_index_stack):  # 这里应该要深度优先遍历完整个子树，而不是只遍历一个子支
                        match = False
                        
                        # 直到右节点的name_index对应的节点已经找到左节点的某一条和他完全匹配下来（不包括左节点那一条的叶子节点）
                        # 此时temp_index代表左支与右支完全匹配后，和右支同一深度的节点的idx（不一定下面就是叶子节点，也可能tmp_idx下面还是一些子树）
                        if name_index + 1 > len(name_list):
                            children2 = []
                            for c in program[temp_index].child_distance_list:  # 遍历子节点
                                # children2要加上对pow的检测
                                # children2记录左边子树相同深度层的变量节点和pow的指数节点
                                if not isinstance(program[temp_index + c], tuple):
                                    if isinstance(program[temp_index], _Function) and \
                                            program[temp_index].name == 'pow':  # 父节点是pow函数
                                        # pow函数的最后一个操作数，即指数节点
                                        if c == program[temp_index].child_distance_list[-1]:
                                            children2.append(program[temp_index + c])
                                        else:
                                            # 如果此时是pow的第一个操作数，并且还是一个function节点呢？
                                            # if not isinstance(program[temp_index + c], tuple)保证进来的时候，如果是pow，左节点进来的就不是变量节点
                                            # 就默认不会抵消吗
                                            no_cancel = True
                                            break
                                    else:
                                        # tmp_index是非pow的函数节点，并且孩子不是变量节点,则认为不抵消？(因为抵消概率很小？)
                                        no_cancel = True
                                        break
                                else:
                                    # 如果temp_index是pow，并且他的左孩子是变量节点，那么会进来这个分支
                                    children2.append(program[temp_index + c])
                            if no_cancel:  # 子节点有常数或函数节点就认为不会抵消
                                break
                            if not len(children):  # children为空，说明当前函数arity为1
                                # 说明当前temp_index节点 (与current_index的父节点函数名相同)是arity为1的函数节点
                                # children装的就是当前节点父节点的其他孩子节点
                                # 直接将children2中唯一一个元素记录到prohibit中
                                prohibit.append(children2[0])
                                break
                            if program[temp_index].name == program[parent_index].name:
                                if program[temp_index].name in ['add', 'mul', 'max', 'min',
                                                                'sum', 'prod', 'mean']:
                                    for c in children:
                                        if c in children2:  # 在children2中，则将children2中相同的节点去除
                                            children2.remove(c)
                                        if c == children[-1] and len(children2) == 1:
                                            prohibit.append(children2[0])
                                else:  # 其他函数则进行序列匹配
                                    for c_i, c in enumerate(children):
                                        if c != children2[c_i]:  # # 如果有一个不相同，则不会抵消
                                            break
                                        if c == children[-1]:  # 前面都相同，则禁止生成children2的最后一个
                                            prohibit.append(children2[-1])
                            break
                        temp_name = name_list[- name_index - 1]
                        fence = child_index_stack.pop()  # 获取最新的
                        temp_children = program[temp_index].child_distance_list
                        for j, c in enumerate(temp_children):
                            if j < fence:
                                continue
                            if isinstance(program[temp_index + c], _Function) and \
                                    program[temp_index + c].name == temp_name:  # 找到了一个匹配的子节点
                                child_index_stack.append(j + 1)  # 记录当前层的下一个未探索的子节点索引
                                temp_index = temp_index + c
                                child_index_stack.append(0)  # 记录下一层第一个未探索的子节点索引
                                name_index += 1  # 搜索下一个名字
                                match = True
                                break
                        if not match:  # 这个当前点的子节点没有匹配的，则返回至当前点的父节点
                            temp_index = temp_index + program[temp_index].parent_distance  # 确保初始值正确
                            name_index -= 1  # 搜索上一个名字
            else:  # 与sub之间没有中间函数节点
                if isinstance(program[temp_index], tuple):
                    prohibit.append(program[temp_index])
        return prohibit

    def calculate_value_range(self, program, parent_index, parent_name):
        temp_range = np.array([0., 0.])
        arity = program[parent_index].arity
        if parent_name == 'mean':
            # 遍历mean的所有孩子节点
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                # 如果这个孩子是函数节点
                if isinstance(program[parent_index + child_distance], _Function):
                    # 直接拿孩子函数节点的范围
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    # 使用变量的取值范围
                    span = np.array(self.variable_range)
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
            # 叠加每个孩子的取值情况
                temp_range += span
            temp_range = np.array([temp_range[0] / arity, temp_range[1] / arity])
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name in ['add', 'sum']:
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
                # print(f'add span:{span}')
                # add和sum的逻辑和mean差不多，最后不需要做平均
                temp_range += span
                # print(f'add temp_range:{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'sub':
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
                # print(f'sub span:{span}')
                # 通过下面的操作实现相减的范围
                if i != 0:  # 第二个操作数
                    span = np.array([-span[1], -span[0]])
                temp_range += span
                # print(f'sub temp_range:{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name in ['mul', 'prod']:
            temp_range = [1, 1]
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
                # print(f'mul span:{span}')
                # 乘法的范围确定需要考虑子节点相乘的组合情况，最终确定
                temp = np.array([temp_range[0] * span[0],
                                 temp_range[0] * span[1],
                                 temp_range[1] * span[0],
                                 temp_range[1] * span[1]])
                temp_range = np.array([np.min(temp), np.max(temp)])
                # print(f'mul temp_range:{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name in ['div', 'inv']:
            temp_range = [1, 1]
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
                if i != 0:  # 第二个操作数
                    # 实际上是对除数的span取值范围进行判断
                    # 看其位于[-1e3,1e3]的哪个位置，要进行对应的取值处理
                    if span[0] <= -0.001 and 0.001 <= span[1]:
                        span = [-1000, 1 / span[0], 1 / span[1], 1000]
                    elif -0.001 <= span[0] <= 0.001 <= span[1]:
                        span = [1 / span[1], 1000]
                    elif 0.001 <= span[0] or span[1] <= -0.001:
                        span = [1 / span[1], 1 / span[0]]
                    elif span[0] <= -0.001 <= span[1] <= 0.001:
                        span = [-1000, 1 / span[0]]
                    elif -0.001 <= span[0] <= span[1] <= 0.001:
                        span = [1, 1]
                if len(span) < 4:
                    temp = np.array([temp_range[0] * span[0],
                                     temp_range[0] * span[1],
                                     temp_range[1] * span[0],
                                     temp_range[1] * span[1]])
                else:
                    temp = []
                    for m in span:
                        for n in temp_range:
                            temp.append(m * n)
                    temp = np.array(temp)
                # print(f'div span:{span}')
                temp_range = np.array([np.min(temp), np.max(temp)])
                # print(f'div temp_range:{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'pow':
            # if isinstance(program[-2], _Function):
            #     span = program[-2].value_range
            # else:
            #     span = np.array(self.variable_range)
            # for i, child_distance in enumerate(program[parent_index].child_distance_list):
            first_child_distance = program[parent_index].child_distance_list[0]  # pow的第一个操作数
            if isinstance(program[parent_index + first_child_distance], _Function):
                span = program[parent_index + first_child_distance].value_range
            elif isinstance(program[parent_index + first_child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + first_child_distance][0]))
                span = np.array([-max_num, max_num])
            else:
                span = np.array(self.variable_range)
            span = np.array(span, dtype=np.float64)
            second_child_distance = program[parent_index].child_distance_list[1]  # pow的第二个操作数
            exponent = program[parent_index + second_child_distance][0][0]  # 取指数向量中的第一个
            # print(f'pow span:{span}, exponent:{exponent}')
            if exponent > 0:
                if exponent % 2 == 0:
                    if span[0] >= 0 or span[1] <= 0:
                        x = np.power(span, exponent)
                        temp_range = np.array([np.min(x), np.max(x)])
                    else:
                        temp_range = np.array([0, np.max(np.power(span, exponent))])
                else:  # 指数>0且为奇数，单调递增
                    temp_range = np.power(span, exponent)
            else:  # 指数小于0，涉及分式，-0.001和0.001为临界点
                if exponent % 2 == 0:
                    if span[0] <= -0.001 and 0.001 <= span[1]:
                        temp_range = np.array([np.min(np.power(span, exponent)), 0.001 ** exponent])
                    elif -0.001 <= span[0] <= 0.001 <= span[1]:
                        temp_range = np.power([span[1], 0.001], exponent)
                    elif 0.001 <= span[0] or span[1] <= -0.001:
                        x = np.power(span, exponent)
                        temp_range = np.array([np.min(x), np.max(x)])
                    elif span[0] <= -0.001 <= span[1] <= 0.001:
                        temp_range = np.power([span[0], 0.001], exponent)
                    elif -0.001 <= span[0] <= span[1] <= 0.001:
                        temp_range = [-1, 1]  # 无效
                else:  # 指数<0且为奇数，这里为了方便某些情况只管右支或近似估计
                    if span[0] <= -0.001 and 0.001 <= span[1]:
                        temp_range = np.power([-0.001, 0.001], exponent)
                    elif -0.001 <= span[0] <= 0.001 <= span[1]:
                        temp_range = np.power([span[1], 0.001], exponent)
                    elif 0.001 <= span[0] or span[1] <= -0.001:
                        x = np.power(span, exponent)
                        temp_range = np.array([np.min(x), np.max(x)])
                    elif span[0] <= -0.001 <= span[1] <= 0.001:
                        temp_range = np.power([-0.001, span[0]], exponent)
                    elif -0.001 <= span[0] <= span[1] <= 0.001:
                        temp_range = [-1, 1]  # 无效
            # 实际上这里理论上也会出现inf的valuerange
            # print(f'pow temp_range:{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'max':
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
                if i == 0:
                    temp_range = span
                else:
                    temp_range = np.array([np.max([temp_range[0], span[0]]),
                                           np.max([temp_range[1], span[1]])])
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'min':
            for i, child_distance in enumerate(program[parent_index].child_distance_list):
                if isinstance(program[parent_index + child_distance], _Function):
                    span = program[parent_index + child_distance].value_range
                elif isinstance(program[parent_index + child_distance], list):  # 取常数向量中最大的数字来生成对称区间
                    max_num = np.max(np.abs(program[parent_index + child_distance][0]))
                    span = np.array([-max_num, max_num])
                else:
                    span = np.array(self.variable_range)
            # for i in range(arity):
            #     if isinstance(program[- 1 - i], _Function):
            #         span = program[- 1 - i].value_range
            #     else:
            #         span = np.array(self.variable_range)
                if i == 0:
                    temp_range = span
                else:
                    temp_range = np.array([np.min([temp_range[0], span[0]]),
                                           np.min([temp_range[1], span[1]])])
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name in ['sin', 'cos', 'tan','tanh']:  # 三角函数
            temp_range = np.array([-1, 1])
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'exp':
            # if isinstance(program[-1], _Function):
            #     span = program[-1].value_range
            # else:
            #     span = np.array(self.variable_range)
            if isinstance(program[parent_index + 1], _Function):
                span = program[parent_index + 1].value_range
            elif isinstance(program[parent_index + 1], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + 1][0]))
                span = np.array([-max_num, max_num])
            else:
                span = np.array(self.variable_range)
            # temp_range = np.exp(span)
            # 之前没有闭包的时候，取值范围出现inf应该就是这里出问题
            temp_range = _protected_exp(span)
            if not validate_interval(temp_range):
                # print(f'name:{parent_name}, value_range:{temp_range}')
                temp_range = np.array([temp_range[0], temp_range[0] + 1000])
        elif parent_name == 'neg':
            # if isinstance(program[-1], _Function):
            #     span = program[-1].value_range
            # else:
            #     span = np.array(self.variable_range)
            if isinstance(program[parent_index + 1], _Function):
                span = program[parent_index + 1].value_range
            elif isinstance(program[parent_index + 1], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + 1][0]))
                span = np.array([-max_num, max_num])
            else:
                span = np.array(self.variable_range)
            temp_range = np.array([-span[1], -span[0]])
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'log':
            # if isinstance(program[-1], _Function):
            #     span = program[-1].value_range
            # else:
            #     span = np.array(self.variable_range)
            if isinstance(program[parent_index + 1], _Function):
                span = program[parent_index + 1].value_range
                # if span[1] <= span[0]:
                #     print(f'span1:{span}')
            elif isinstance(program[parent_index + 1], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + 1][0]))
                span = np.array([-max_num, max_num])
                # if span[1] <= span[0]:
                #     print(f'span2:{span}')
            else:
                span = np.array(self.variable_range)
                # if span[1] <= span[0]:
                #     print(f'span3:{span}')
            if span[0] <= 0 <= span[1]:
                temp_range = np.log([0.001, np.max(np.abs(span))])
                # if np.array_equal(temp_range, [0.0, 0.0]):
                #     print(f'log1:{span}{temp_range}')
            elif span[1] <= 0:
                temp_range = np.log(np.abs([np.min([-0.001, span[1]]), span[0]]))
                # if np.array_equal(temp_range, [0.0, 0.0]):
                #     print(f'log2:{span}{temp_range}')
            elif span[0] >= 0:
                temp_range = np.log([np.max([0.001, span[0]]), span[1]])
                # if np.array_equal(temp_range, [0.0, 0.0]):
                #     print(f'log3:{span}{temp_range}')
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'sqrt':
            # if isinstance(program[-1], _Function):
            #     span = program[-1].value_range
            # else:
            #     span = np.array(self.variable_range)
            if isinstance(program[parent_index + 1], _Function):
                span = program[parent_index + 1].value_range
            elif isinstance(program[parent_index + 1], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + 1][0]))
                span = np.array([-max_num, max_num])
            else:
                span = np.array(self.variable_range)
            if span[0] <= 0 <= span[1]:
                temp_range = np.sqrt([0, np.max(np.abs(span))])
            elif span[1] <= 0:
                temp_range = np.sqrt(np.abs([span[1], span[0]]))
            elif span[0] >= 0:
                temp_range = np.sqrt(span)
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        elif parent_name == 'abs':
            # if isinstance(program[-1], _Function):
            #     span = program[-1].value_range
            # else:
            #     span = np.array(self.variable_range)
            if isinstance(program[parent_index + 1], _Function):
                span = program[parent_index + 1].value_range
            elif isinstance(program[parent_index + 1], list):  # 取常数向量中最大的数字来生成对称区间
                max_num = np.max(np.abs(program[parent_index + 1][0]))
                span = np.array([-max_num, max_num])
            else:
                span = np.array(self.variable_range)
            if span[0] <= 0 <= span[1]:
                temp_range = np.array([0, np.max(np.abs(span))])
            elif span[1] <= 0:
                temp_range = np.abs([span[1], span[0]])
            elif span[0] >= 0:
                temp_range = span
            # if not validate_interval(temp_range):
            #     print(f'name:{parent_name}, value_range:{temp_range}')
        else:
            raise ValueError('No function matched.')
        # if temp_range[1] >= 1e3:
        #     temp_range[1] = 1e3
        # if temp_range[0] <= -1e3:
        #     temp_range[0] = -1e3
        # temp_range = np.array(temp_range, dtype=np.int32)
        temp_range = np.array(temp_range, dtype=np.float64)
        if temp_range[1] <= temp_range[0]:
            if np.abs(temp_range[1] - temp_range[0]) <= 1:
                temp_range = np.array([temp_range[1], temp_range[1] + 1], dtype=np.float64)
            else:
                raise ValueError(f'temp_range:{parent_name}{temp_range}')
        return temp_range

    @staticmethod
    def subtree_state_larger(remaining, total):  # 比较remaining和total，当第二个被第一个dominate时返回true
        temp = [remaining[i] - total[i] for i in range(len(remaining))]
        return temp[0] >= 0 and temp[1] >= 0 and temp[2] >= 0 and temp[3] >= 0  # 次数都有剩余，则可以兼容

    # @staticmethod
    # def check_prohibit(parent_name, node):  # 给定父节点名字和当前节点，检查是否会导致违规的连续嵌套
    #     if not isinstance(node, _Function):  # 当前节点不是函数节点则没有限制，返回True
    #         return True
    #     elif parent_name == 'exp' and node.name == 'log':
    #         return False
    #     elif parent_name == 'log' and node.name == 'exp':
    #         return False
    #     elif parent_name == node.name and parent_name in ['abs', 'neg', 'add', 'sum', 'sqrt']:
    #         return False
    #     elif parent_name in elementary_functions and isinstance(node, list):  # 父节点是基本初等函数，则子节点不可以只是常数
    #         return False
    #     return True

    @staticmethod
    def get_max_depth(program):  # 求树的深度，即求层数
        terminal_stack = []
        max_depth, current_depth = 0, 0
        for i, node in enumerate(program):
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
                current_depth += 1
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    max_depth = max(max_depth, current_depth)  # 记录节点的最大深度，节点的最大深度就是树的深度
                    current_depth -= 1
                    terminal_stack.pop()
                    if not terminal_stack:  # 遍历结束后返回树的深度
                        return max_depth
                    terminal_stack[-1] -= 1

    def printout(self, program):
        for node in program:
            if isinstance(node, _Function):
                if node.name in ['sum', 'prod', 'mean']:
                    print(f"{node.name}[{node.input_dimension},{node.arity}]",
                          end=' ')  # ,{node.parent_distance},{node.child_distance_list}
                else:
                    print(f"{node.name}[{node.input_dimension}]",
                          end=' ')  # [{node.parent_distance},{node.child_distance_list}]
            elif isinstance(node, tuple):  # 变量节点
                print(f'{node[0],self.n_features - node[1], node[2]}', end=' ')
            else:
                print(node, end=' ')
        print()

    def print_formula(self, program, show_operand=False):  # 颜色编码从'\033[31m'到'\033[38m'
        formula_stack = []
        min_priority_stack = []  # 用于子树的内括号判断，记录每个子树的min_priority
        # name_mapping = {'add': '+', 'sub': '-', 'mul': '×', 'div': '/', 'pow': '^'}  # , 'sum': '+', 'prod': '×'
        name_mapping = {'add': '+', 'neg': '-', 'sub': '-', 'mul': '×', 'div': '/',
                        'pow': '^'}  # , 'sum': '+', 'prod': '×'
        # priority = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
        priority = {'add': 1, 'neg': 2, 'sub': 2, 'mul': 3, 'div': 4, 'pow': 5}  # 操作符的优先级
        formula = ''
        last_arity = 0
        last_name = ''
        min_priority = 5  # 用于子树的外括号判断
        for node in program:
            if isinstance(node, _Function):
                formula_stack.append(node.name)
                formula_stack.append(node.arity)
                last_name = node.name
                last_arity = node.arity
            else:
                if show_operand:  # 展示具体的操作数，操作数分为向量切片(tuple)和常数向量list[ndarray]两种
                    temp = '\033[36m' + '[' + '\033[0m'
                    if isinstance(node, tuple):
                        for i in range(node[0], self.n_features - node[1], node[2]):
                            temp += 'X' + str(i) + ', '
                    else:
                        for i in node[0]:  # 遍历ndarray
                            temp += str(i) + ', '
                    temp = temp[:-2] + '\033[36m' + ']' + '\033[0m'
                    formula_stack.append(temp)
                else:
                    formula_stack.append('o')
                # 如果arity已经满足，且中间没有arity数字，说明操作数数目已经满足
                # 同时在该完整子树内进行内括号判断，即根据基本操作符的优先级顺序来添加括号
                while last_arity + 1 <= len(formula_stack) and \
                        formula_stack[-(last_arity + 1)] == last_arity and \
                        formula_stack[-(last_arity + 1)] not in formula_stack[- last_arity:]:
                    for i in range(last_arity):  # 移除末尾last_arity个操作数
                        intermediate = formula_stack.pop()
                        if intermediate[0] != '@':  # 不是子树，则不需要内括号判断
                            formula = intermediate + formula
                            if last_name == 'neg':
                                min_priority = 0  # neg对外优先级最低，对内优先级与sub相同
                            elif last_name in name_mapping.keys():
                                min_priority = priority[last_name]
                            else:
                                min_priority = 5
                        else:  # 如果是子树，则需要知道min_priority
                            intermediate = intermediate[1:]  # 去掉第一个特殊字符@
                            min_priority = min_priority_stack.pop()  # 获取最后一个子树的min_priority
                            if last_name in name_mapping.keys():  # 在函数名字-符号映射表内，则判断是否需要添加括号
                                if min_priority < priority[last_name] or \
                                        min_priority == priority[last_name] and min_priority % 2 == 0:  # 相等且为2或4，即减和除
                                    formula = '(' + intermediate + ')' + formula
                                else:
                                    formula = intermediate + formula
                                if last_name != 'neg':
                                    min_priority = priority[last_name]  # 优先级取最低的
                                else:
                                    min_priority = 0
                            else:  # 其他函数，无需内括号判断
                                formula = intermediate + formula
                                min_priority = 5
                        if i != last_arity - 1:  # 不是最后一个操作数，则需要加上操作符
                            if last_name in name_mapping.keys():
                                formula = name_mapping[last_name] + formula
                            else:
                                formula = ', ' + formula
                        elif last_name == 'neg':
                            formula = name_mapping[last_name] + formula
                    formula_stack.pop()  # 移除函数节点的arity数字
                    formula_stack.pop()  # 移除函数节点的函数名字
                    # 一个完整子树的外括号判断：abs为||；neg为-，并且判断是否需要添加括号；非基本操作符的函数外括号添加，聚集函数使用{}，其余使用()
                    if last_name == 'abs':
                        front = '\033[32m' + '|' + '\033[0m'
                        end = '\033[32m' + '|' + '\033[0m'
                        formula = front + formula + end
                    # elif last_name == 'neg':
                    #     if min_priority <= 2:
                    #         formula = '-(' + formula + ')'
                    #     else:
                    #         formula = '-' + formula
                    #     min_priority = 0
                    elif last_name not in name_mapping.keys():  # 若不在函数名字-符号映射表内，则需要加上函数节点的名字
                        if last_name in aggregate:
                            front = '\033[31m' + '{' + '\033[0m'
                            end = '\033[31m' + '}' + '\033[0m'
                            formula = last_name + front + formula + end
                        else:
                            front = '('
                            end = ')'
                            formula = last_name + front + formula + end
                    if len(formula_stack) == 0:  # formula为空
                        print(formula)
                        return
                    # 找新的最后一个函数节点，更新last_name和last_arity
                    for index in range(len(formula_stack)):
                        if not isinstance(formula_stack[- 1 - index], str):  # 不是str，则为arity
                            last_arity = formula_stack[- 1 - index]
                            last_name = formula_stack[- 2 - index]
                            break
                    formula = '@' + formula  # @开头表示这是一个子树的文本表示
                    formula_stack.append(formula)  # 附加到末尾
                    min_priority_stack.append(min_priority)
                    formula = ''
                    min_priority = 5

    def check_subtree_identity(self, root_div, left_subtree, right_subtree):  # 左右子树相同的判断方法
        left_subtree = self.remove_neg_abs(root_div=root_div, subtree=left_subtree)
        right_subtree = self.remove_neg_abs(root_div=root_div, subtree=right_subtree)
        return self.check_subtree_identity_recursively(left_subtree, right_subtree)

    def remove_neg_abs(self, root_div, subtree):
        """
        div和sub的递归基础情况不一样，div需要忽略可以忽略的neg和abs节点。
        abs节点可以通过分类讨论转化为正负符号，与neg节点的效果类似。
        对于neg和abs节点：
        - 若一个函数的arity为1且其为奇偶函数，则可以忽略其直接neg和abs子节点。
          这类函数有sqrt(protected)、log(protected)、neg、abs、sin、cos、tan、inv。
          arity为1的函数节点仅exp不是奇偶函数，不可忽略直接neg和abs子节点。
        - 若一个函数的arity>1：
          - 若其关于每个变量都有奇偶性，则也可以忽略直接neg和abs子节点。
            这类函数有prod、mul、div、pow(pow实际上是单节点函数，指数位置必定是常数函数)。
          - 若其关于每个变量不具有奇偶性，则不可忽略直接neg和abs子节点。
            这类函数有sum、mean、add、sub、max、min。
            对于这一类函数，需要额外检查左右子树是否互为相反数
        奇(偶)=偶，奇(奇)=奇，偶(奇)=偶，偶(偶)=偶 -> 奇偶函数的复合函数依旧是奇偶函数
        sub为根节点则只需要判断子树对应节点是否完全相同就行。
        但还存在比较明显的同质函数节点问题。比如对div来说，直接sum和mean子节点其实是同质的，仅系数不同。
        对div来说，同质的函数节点有(sum,mean,add)、(prod,mul)。但目前先不引入这个概念。

        这个函数会修改原来的program，将其中对语义没有影响的neg和abs节点去除。
        """
        subtree = deepcopy(subtree)
        neg_abs_to_delete = []  # 记录要删除的neg和abs节点的索引
        if root_div:
            for index, item in enumerate(subtree):  # 遍历所有节点，将奇偶函数下的直接neg和abs节点都记录下来
                if isinstance(item, _Function) and item.name in ['neg', 'abs']:
                    # 这里subtree的根节点并不是真正意义上的根节点，parent_distance不为0，depth也不为0。
                    # 函数要增加一个参数，表明左右子树的父节点是否奇偶函数
                    if index == 0:
                        # 若父节点是奇偶函数，则可以忽略该neg和abs节点
                        neg_abs_to_delete.append(index)
                    else:  # 不是左子树的根节点，则有父节点
                        parent = subtree[index + item.parent_distance]
                        if parent.name in ['sqrt', 'log', 'neg', 'abs', 'sin', 'cos',
                                           'tan', 'inv', 'prod', 'mul', 'div', 'pow','tanh']:  # 奇偶函数名字列表
                            neg_abs_to_delete.append(index)
        else:
            for index, item in enumerate(subtree):  # 遍历所有节点，将奇偶函数下的直接neg和abs节点都记录下来
                if isinstance(item, _Function) and item.name in ['neg']:
                    # sub也可以忽略偶函数的直接neg子节点，特别的，neg(neg)嵌套可以一起去除
                    if index != 0:  # 不是左子树的根节点，则有父节点
                        parent = subtree[index + item.parent_distance]
                        if parent.name in ['sqrt', 'log', 'abs', 'cos']:  # 偶函数名字列表，pow奇偶性与指数有关，这里就不关注
                            neg_abs_to_delete.append(index)
                        elif parent.name == 'neg':  # 父节点是neg，则父节点的索引是index - 1
                            if (index - 1) not in neg_abs_to_delete:  # 父节点neg仍生效才能一起加入neg_abs_to_delete中
                                neg_abs_to_delete.append(index - 1)
                                neg_abs_to_delete.append(index)
        # print()
        # print('program before pop:', end='')
        # self.printout(program=subtree)

        # 根据neg_abs_to_delete删减subtree中可忽略的neg和abs节点
        offset = -1
        for index in neg_abs_to_delete[::-1]:  # 要反着来删除节点，否则会引起删除的语义错误
            fence = 0  # 从根节点开始更新parent_distance和child_distance_list属性
            while fence < index:
                for i, distance in enumerate(subtree[fence].child_distance_list[::-1]):
                    if fence + distance > index:  # 位于index后的子节点
                        # 该子节点是函数节点，则更新parent_distance
                        if isinstance(subtree[fence + distance], _Function):
                            subtree[fence + distance].parent_distance -= offset
                        subtree[fence].child_distance_list[- 1 - i] += offset
                    else:  # fence + distance <= index
                        fence = fence + distance
                        break
            parent_distance = subtree[index].parent_distance
            subtree.pop(index)
            if isinstance(subtree[index], _Function):
                subtree[0].parent_distance = parent_distance

            # # 若abs和neg节点的直接子节点是函数节点，则需要调整parent_distance
            # parent_distance = subtree[index].parent_distance  # abs和neg节点的parent_distance
            # subtree.pop(index)
            # # print('program after pop :', end='')
            # # self.printout(program=subtree)
            # if isinstance(subtree[index], _Function):
            #     subtree[index].parent_distance = parent_distance
            #     # print(index)
            #     # print(parent_distance)
            # # 删除abs和neg节点的同时还需要调整受到影响的parent_distance和child_distance_list属性
            # # 获取这个abs或neg节点的子树，在这个子树后的函数节点需要调整parent_distance和child_distance_list属性
            # temp_subtree = self.get_subtree(root=index, program=subtree)
            # root_index = index + len(temp_subtree)  # 后面所有子树的根节点的parent_distance属性都要调整
            # while root_index < len(subtree):  # 存在该分支
            #     root = subtree[root_index]
            #     parent_index = root_index + root.parent_distance
            #     # if not isinstance(subtree[parent_index], _Function):
            #     #     print('program before assert:', end='')
            #     #     self.printout(program=subtree)
            #     #     print(neg_abs_to_delete)
            #     #     print(root_index)
            #     #     print(root.parent_distance)
            #     #     print(parent_index)
            #     assert isinstance(subtree[parent_index], _Function)
            #     for i, child_distance in enumerate(subtree[parent_index].child_distance_list):
            #         if parent_index + child_distance == root_index:
            #             subtree[parent_index].child_distance_list[i] -= 1
            #             break
            #     root.parent_distance += 1
            #     temp_subtree = self.get_subtree(root=root_index, program=subtree)
            #     root_index = root_index + len(temp_subtree)  # 后面所有子树的根节点的parent_distance属性都要调整
        return subtree

    def check_subtree_identity_recursively(self, left_subtree, right_subtree):  # 左右子树相同的判断方法
        if len(left_subtree) != len(right_subtree):  # 两边长度不相同还子树相同是小概率事件
            return False
        elif len(left_subtree) == 1:  # 两边长度相同且为1
            left, right = left_subtree[0], right_subtree[0]
            if isinstance(left, tuple) and left == right:  # 两个变量节点相同，返回True
                return True
            else:
                return False
        else:  # 两边长度相同但不是1，根节点一定是函数节点
            # 首先判断根节点是否相同
            left, right = left_subtree[0], right_subtree[0]
            # 这里名字相同可以改成同质函数判断，但同质性还跟中间函数节点是否和根节点性质相同有关
            while left.name == right.name and left.arity == right.arity and left.arity <= 2:  # 函数名相同，判断各有多少子树
                # arity相同且不大于2，大于2的情况还相同的情况是小概率事件
                if left.arity == 1:  # arity为1时可以直接通过循环判断
                    left_subtree, right_subtree = left_subtree[1:], right_subtree[1:]
                    left, right = left_subtree[0], right_subtree[0]  # 下一根节点
                    if isinstance(left, tuple) and left == right:  # 两个变量节点相同，返回True
                        return True
                    elif not (isinstance(left, _Function) and isinstance(right, _Function)):
                        return False
                else:  # left.arity == 2
                    left_children = left.child_distance_list  # 子节点相对于当前点的距离就是在left_subtree中的坐标
                    right_children = right.child_distance_list
                    if left.name in ['add', 'mul', 'max', 'min', 'sum', 'prod', 'mean']:  # 子树组合匹配
                        forward = True
                        inverse = True
                        # 2的组合数是2，正向和逆向匹配
                        for i in range(left.arity):  # 正向匹配
                            forward = forward and self.check_subtree_identity_recursively(
                                left_subtree=self.get_subtree(
                                    root=left_children[i],
                                    program=left_subtree),
                                right_subtree=self.get_subtree(
                                    root=right_children[i],
                                    program=right_subtree))
                            if not forward:  # forward为False就可以停止对forward的计算了
                                break
                        for i in range(left.arity):  # 逆向匹配
                            inverse = inverse and self.check_subtree_identity_recursively(
                                left_subtree=self.get_subtree(
                                    root=left_children[i],
                                    program=left_subtree),
                                right_subtree=self.get_subtree(
                                    root=right_children[- i - 1],
                                    program=right_subtree))
                            if not inverse:  # inverse为False就可以停止对inverse的计算了
                                break
                        return forward or inverse  # 两者有一个为True即子树相同
                    else:  # 子树排列匹配
                        forward = True
                        for i in range(left.arity):  # 正向匹配
                            forward = forward and self.check_subtree_identity_recursively(
                                left_subtree=self.get_subtree(
                                    root=left_children[i],
                                    program=left_subtree),
                                right_subtree=self.get_subtree(
                                    root=right_children[i],
                                    program=right_subtree))
                            if not forward:  # forward为False就可以停止递归了
                                return False
                        return forward
            else:
                return False

    def get_subtree(self, root, program=None):  # 给定program和指定节点，获取以该节点为根节点的program'
        if program is None:
            program = self.program
        if root >= len(program):
            raise ValueError('root is out of program.')
        terminal_stack = []
        # program = deepcopy(program)  # 深复制，如果subtree需要被修改的话要加上这句话，但目前只删掉不影响语义的abs和neg节点
        if not isinstance(program[root], _Function):  # 不是函数对象直接返回该点
            return [program[root]]  # 类型统一，保持为list类型
        for i, node in enumerate(program[root:]):  # 从root开始
            if isinstance(node, _Function):
                terminal_stack.append(node.arity)
            else:
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program[root:root+i+1]
                    terminal_stack[-1] -= 1



    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
    
    
    def copy_members(self, other_program):
        """将另一个 _Program 对象的成员变量复制到当前对象中"""
        self.function_set = other_program.function_set
        self.arities = other_program.arities
        self.init_depth = other_program.init_depth
        self.mutate_depth = other_program.mutate_depth
        self.init_method = other_program.init_method
        self.n_features = other_program.n_features
        self.variable_range = other_program.variable_range
        self.metric = other_program.metric
        self.p_point_replace = other_program.p_point_replace
        self.parsimony_coefficient = other_program.parsimony_coefficient
        self.transformer = other_program.transformer
        self.feature_names = other_program.feature_names
        self.program = other_program.program
        self.problemID = other_program.problemID
        self.problem_coord = other_program.problem_coord
        self.model = other_program.model
        self.scaler = other_program.scaler
        self.save_path = other_program.save_path
        self.best_dim = other_program.best_dim
        self.raw_fitness_ = other_program.raw_fitness_
        self.fitness_ = other_program.fitness_
        self.parents = other_program.parents
        self._n_samples = other_program._n_samples
        self._max_samples = other_program._max_samples
        self._indices_state = other_program._indices_state
        self.late_expr = other_program.latex_expr
        self.latex_expr_with_const = other_program.latex_expr_with_const
        self.eval_expression = other_program.eval_expression

