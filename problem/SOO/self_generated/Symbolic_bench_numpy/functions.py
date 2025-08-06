"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, input_dimension=0, output_dimension=0, depth=0):  # 除了function，其他都可视作实例属性
        self.function = function
        self.name = name
        self.arity = arity  # arity是一个实例属性，可以每个实例都不一样，用于拓展累加和累乘运算符
        # 用于保证公式中操作数维度一致
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        # 以上的属性new_operator函数会维护，以下这些属性在point_mutation中需要手动维护
        self.depth = depth  # 用于寻找父节点
        # 以下两个属性统称子树状态
        ''' 四元组(剩余aggregate权重(sum、prod、mean权重为2)，
         剩余pow权重,剩余基本初等函数权重,剩余exp次数)，用于控制复杂度，缩小搜索空间'''
        self.remaining = [4, 1, 1, 1]  # 默认值为[4, 1, 1, 1]。remaining包含了当前点以及祖先有多少特殊节点的信息
        # 四元组(已有aggregate权重(sum、min和max权重为1，prod、mean权重为3)，已有pow权重，已有基本初等函数权重，已有exp次数)，用于控制复杂度，缩小搜索空间
        self.total = [0, 0, 0, 0]  # 默认值为[0, 0, 0, 0]。total包含了以该节点作为根节点的子树总共有多少特殊节点的信息
        self.constant_num = 0  # 计数有多少个子节点是常数，限制最多只能有一个常数子节点
        self.value_range = np.array([])  # 值域
        # 用相对距离来记录父节点和子节点的位置，这样可以避免交叉突变和hoist突变带来的维护成本
        self.parent_distance = 0  # 记录父节点相对于自己的距离，除了根节点的其他节点该属性应当是个负数
        self.child_distance_list = []  # 记录各个子节点相对于自己的距离
        # self.branch_set = []  # 用于sub和div的抵消检测

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):  # np.ufunc是universal function的缩写，指这些函数能作用域ndarray的每个元素上，而不是对ndarray对象操作
        if function.__code__.co_argcount != arity:  # 检测函数参数数目与定义的arity是否相同
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]  # 构造arity个10维向量作为检测点，检测输出的维度
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):  # 此处无需修改，只需修改build_program中对sum或prod的操作数的选择即可
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):  # 有输出值是无穷
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):  # 负输入的输出合法性检测
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # 在该环境下，除以零和无效值0/0不报错，均无视
        result = np.where(np.abs(x2) > 0.0001, np.divide(x1, x2), 1.)
        # result = np.where(result >= 1e50, 1e50, result)
        return result


# def _protected_multiply(x1, x2):
#     """Closure of multiply (x1/x2) for large numbers."""
#     with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # 在该环境下，除以零和无效值0/0不报错，均无视
#         # result = np.where(np.abs(x2) < 1e50, np.multiply(x1, x2), 1.)
#         # result = np.where(result > 1e50, 1e50, result)
#         return np.multiply(x1, x2)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.0001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


# 当底数X为负数时，指数c必须是整数类型才能保证闭包，否则会遇到nan；当底数X为整数时，指数c必须是非负数，否则ValueError
# 这里保证输入X是浮点数，所以指数c是整数类型(int32)即可，且当底数X的绝对值接近于0时返回0.0(对于指数为正整数的情况还算合理)，底数为0且指数为负数时会出现inf
def _protected_power(X, c):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        # 要对c进行正负判断
        # 防inf处理
        result =  np.where((np.abs(X) > 0.0001) | (np.array(c) >= 0), np.power(np.array(X, dtype=np.float64), c), 1.)
        return result

def _protected_sub(x1, x2):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.subtract(x1, x2)


def _protected_exp(X):
    with np.errstate(over='ignore', invalid='ignore'):
        # return np.where(X > 50, 1.0, np.exp(X))
        return np.exp(X)



def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


def _sum(X):  # 累加，arity定义为0，但构造公式时应该赋予一个随机生成的值作为元数
    return X.sum(axis=0)


def _prod(X):  # 累乘，arity定义为0，但构造公式时应该赋予一个随机生成的值作为元数
    return X.prod(axis=0)


def _mean(X):  # 求平均值，arity定义为0，但构造公式时应该赋予一个随机生成的值作为元数
    return X.mean(axis=0)

def _tanh(x1):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return (_protected_exp(x1) - _protected_exp(-x1))/(_protected_exp(x1) + _protected_exp(-x1))

# 所有函数均基于np实现。函数约束规则：不能嵌套
sum = _Function(function=_sum, name='sum', arity=0)
prod = _Function(function=_prod, name='prod', arity=0)
mean = _Function(function=_mean, name='mean', arity=0)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
tanh1 = _Function(function=_tanh, name='tanh', arity=1)
exp1 = _Function(function=_protected_exp, name='exp', arity=1)  # 指数函数
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)
add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=_protected_sub, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
pow2 = _Function(function=_protected_power, name='pow', arity=2)  # 指数应该为整数(数组)，范围为{-3, -2, 2, 3}

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'tanh':tanh1,
                 'pow': pow2,
                 'sum': sum,
                 'prod': prod,
                 'mean': mean,
                 'exp': exp1}


if __name__ == "__main__":
    import itertools
    a = ['abs', 'neg']
    b = ['c', 'd']
    a = list(itertools.chain.from_iterable(a))
    print(list(itertools.chain.from_iterable([a, b])))

