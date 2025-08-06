"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import time

import numpy as np
import pandas as pd

from .functions import _Function, _protected_sub, _sum, _prod, _mean, _sigmoid, _protected_power,_tanh
from .functions import _protected_division, _protected_sqrt, _protected_log, _protected_inverse, _protected_exp 

check_constant_function = False

default_remaining = [4, 1, 1, 1]
default_total = [0, 0, 0, 0]
aggregate = ['prod', 'mean', 'sum']  # aggragate函数名称列表
concatenate = ['add', 'sub', 'mul', 'div', 'sum', 'prod', 'mean', 'min', 'max']  # 可能会出现主导问题的连接操作符
operator = ['neg', 'inv', 'abs']  # neg/inv/abs
elementary_functions = ['sin', 'cos', 'tan', 'log','tanh']  # 基本初等函数名称列表
ignore = [['add', 'sub', 'sum'],  # 加性函数节点
          ['add', 'sub', 'sum', 'mul', 'div', 'prod']]  # 加性和乘性函数节点



class _Program(object):

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
        
        
        # if self.program is not None:
        #     if not self.validate_program():
        #         print(self.program)
        #         raise ValueError('The supplied program is incomplete.')
        # else:
        #     # Create a naive random program
        #     self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        
        # 转换成eval和latex表达式的相关变量

        self.latex_expr = None  # 最终的计算表达式
        self.latex_expr_with_const = None
        
        self.eval_expression = None


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

    def execute(self, X):  # X是一个由多个输入向量组成的矩阵
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
        
        # 添加对X的裁剪，避免输入和bestdim不一致的情况
        # if X.ndim == 1:
        #     X = X.reshape[1,-1]
        # if X.shape[-1] != self.best_dim :
        #     if X.shape[-1] > self.best_dim:
        #         X = X[:,:self.best_dim]
        #     else:
        #         raise
        
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
                        # if len(t[0]) < length:  # 做增广
                        #     # TODO 暂时利用t[0]内的最大最小值的取值范围来获得一个粗略的const range
                        #     level = np.max(np.abs(t[0]))
                        #     const_range = (max(level / 10,0.1), math.ceil(level * 10))  # 常数绝对值大小在该范围内即可

                        #     # level = np.max(np.abs(apply_stack[-1][0].value_range))
                        #     # const_range = (max(level / 10,0.1), math.ceil(level * 10))  # 常数绝对值大小在该范围内即可
                            
                        #     if apply_stack[-1][0].name != 'pow':
                        #         temp_t = self.generate_a_terminal(random_state=random_state,
                        #                                       output_dimension=length,
                        #                                       const_range=const_range,
                        #                                       const=True)
                        #     else:
                        #         temp_t = self.generate_a_terminal(random_state=random_state,
                        #                                       output_dimension=length,
                        #                                       const=True,
                        #                                       const_int=True)

                        #     for i, num in enumerate(t[0]):
                        #         temp_t[0][i] = t[0][i]
                        #     t = temp_t  # numpy数组是引用传递，修改t就修改了self.program中对应的常数节点

                        if len(t[0]) > length:  # 做缩减
                            # TODO 计算ela的时候会用到这里
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
    
    def execute_eval(self, x):  # X是一个由多个输入向量组成的矩阵
        self.constants = {}
        self.variables = {}  # 存储变量节点
        if self.eval_expression is None:
            self.eval_expression = self._generate_eval_expression(x,0)
        # 需要根据表达式内的变量和常数来eval对应的数据
        start_time = time.time()
        result = eval(self.eval_expression)
        end_time = time.time()
        # return result, end_time - start_time
        return result
        
        
    def _generate_eval_expression(self, x,index,dimension = None):
        node = self.program[index]
        if isinstance(node, list):  # 常数向量节点
            len_index = len(self.constants)
            const_name = f"C{len_index+1}"
            if dimension == self.n_features - 1:
                self.constants[const_name] = np.array(node[0][:self.best_dim-1])
                self.constants[const_name] = np.repeat([self.constants[const_name]],x.shape[0],axis=0)
            elif dimension == self.n_features:
                self.constants[const_name] = np.array(node[0][:self.best_dim])
                self.constants[const_name] = np.repeat([self.constants[const_name]],x.shape[0],axis=0)
            elif dimension == 1:
                # 此时就是聚合维度
                self.constants[const_name] = np.array(node[0])
            else:
                print(f"dimension error: {dimension},  constance index {index} , node {node[0]}")
            # print(self.constants[const_name].shape)
            return f'np.array({self.constants[const_name].tolist()})'

        if isinstance(node, tuple):  # 变量节点
            a, b, c = node
            # 判断变量节点的类型
            if a == 0:
                if b == 0:
                    var_name = "X"
                elif b == 1 :
                    var_name = "X1"
            else:
                var_name ="X2"
            # var_name = f"x_slice_{index}"
            self.variables[var_name] = f"x[:, {a}:x.shape[1]-{b}:{c}]"
            return f'{self.variables[var_name]}'

        if isinstance(node, _Function):  # 函数节点
            # 递归生成子节点的表达式
            child_expressions = [
                self._generate_eval_expression(x,index + child_distance,node.input_dimension)
                for child_distance in node.child_distance_list
            ]
            # 需要检查node的名称,是否用到的np,以及是否为sum,mean的不同形式,从而进行不同的轴变化
            # 首先check是不是np底层实现
            node_name = node.name
            if node_name == 'mul':
                node_name = 'np.multiply'
            elif node_name == 'neg':
                node_name = 'np.negative'
            elif node_name == 'abs':
                node_name = 'np.abs'
            elif node_name == 'sin':
                node_name = 'np.sin'
            elif node_name == 'cos':
                node_name = 'np.cos'
            elif node_name == 'tanh':
                node_name = '_tanh'
            elif node_name == 'add':
                node_name = 'np.add'
            elif node_name == 'sub':
                node_name = 'np.subtract'
            elif node_name == 'div':
                node_name = '_protected_division'
            elif node_name == 'exp':
                node_name = '_protected_exp'
            elif node_name == 'pow':
                node_name = '_protected_power'
            elif node_name == 'log':
                node_name = '_protected_log'
            elif node_name == 'sqrt':
                node_name = '_protected_sqrt'
                
            # 然后check sum和mean节点是否是arity为1的节点,从而进行不同的操作
            if node_name in ['sum','mean']:
                if node.arity == 1:
                    # 此时的节点为聚合函数
                    # 需要执行对最后一个维度的聚合
                    if node_name == 'sum':
                        return f"np.sum({', '.join(child_expressions)},axis = -1)"
                    else:
                        return f"np.mean({', '.join(child_expressions)},axis = -1)"
                else:
                    # 如果此时不是聚合,而是矩阵直接相加或者平均的做法
                    #其实就直接调用_sum和_mean即可,不过得把他的孩子全部放在一个list里面,然后stack,相当于对第一个维度arity进行求和或者平均
                    # 如果此时不是聚合,而是矩阵直接相加或者平均的做法
                    # 将子节点表达式转换为numpy数组
                    child_expressions = f"np.stack([{', '.join(child_expressions)}])"
                    if node_name == 'sum':
                        return f"_sum({child_expressions})"
                    else:
                        return f"_mean({child_expressions})"
                    
            # 否则就正常执行
            # 组合成函数调用表达式和执行表达式形式
            # print(f"{node.name}({', '.join(child_expressions)})")
            return f"{node_name}({', '.join(child_expressions)})"
    
    def get_latex_expression(self):
        if self.latex_expr is None:
            # 得到初步的preorder expr
            raw_expr = self._generate_latex_expression(0)
            # 递归的解析转换前序遍历
            self.latex_expr = self.preorder_to_latex(raw_expr,self.problemID)
        return self.latex_expr
    
    def get_latex_expr_with_constant(self):
        if self.latex_expr_with_const is None:
            # 得到初步的preorder expr
            raw_expr = self._generate_latex_expression(0)
            # 递归的解析转换前序遍历
            raw_expr = self.preorder_to_latex(raw_expr,self.problemID )
            # 将常数生成到下面
            # 生成常数向量的解释
            constants_explanation = []
            for const_name, const_value in self.latex_constants.items():
                if const_value.shape[-1] != 1:
                    if const_value.shape[-1] == 9:
                        const_value = const_value[:self.best_dim-1]
                    elif const_value.shape[-1] == 10:
                        const_value = const_value[:self.best_dim]
                        
                    const_value_str = np.array2string(const_value, separator=', ')
                    constants_explanation.append(f"{const_name} = {const_value_str}")
            if len(constants_explanation):
                # 将常数解释拼接为一行小字
                constants_explanation_str = "\\noindent \\quad \\text{where : } \n \\begin{quote} \n \(" + "\) \\\\ \( ".join(constants_explanation) + '\)' + "\n \\end{quote}"
            else:
                # 没有常数向量
                constants_explanation_str = " "
            # 返回完整的 LaTeX 字符串
            self.latex_expr_with_const = '\\begin{dmath*}\n' + f"{raw_expr}" + '\n \\end{dmath*}\n' +  f"{constants_explanation_str}"
        return self.latex_expr_with_const
        
    
    def _generate_latex_expression(self, index):
        # 存储latex表达式的常数和变量
        self.latex_constants = {}
        self.latex_variables = {}
        """递归生成表达式"""
        node = self.program[index]
        if isinstance(node, list):  # 常数向量节点
            len_index = len(self.latex_constants)
            # 判断常数向量节点是否为单维度，如果是单维度那么就直接显示
            if np.array(node[0]).shape[-1] == 1:
                const_name = f"{node[0][0]}"
            else:
                const_name = f"C{len_index+1}"
                self.latex_constants[const_name] = np.array(node[0])
            return const_name

        if isinstance(node, tuple):  # 变量节点
            a, b, c = node
            # 判断变量节点的类型
            if a == 0:
                if b == 0:
                    var_name = f"X"
                elif b == 1:
                    var_name = "X1"
            else:
                var_name = "X2"
            self.latex_variables[var_name] = f"x[:, {a}:x.shape[1]-{b}:{c}]"
            return var_name

        if isinstance(node, _Function):  # 函数节点
            # 递归生成子节点的表达式
            child_expressions = [
                self._generate_latex_expression(index + child_distance)
                for child_distance in node.child_distance_list
            ]

            # 对 sum 和 mean 节点进行特殊处理
            if node.name in ['sum', 'mean']:
                if node.arity == 1:
                    # 聚合函数的情况
                    return f"{node.name}({child_expressions[0]})"
                else:
                    # 将多个子节点的结果相加
                    add_expression = self._generate_add_expression(child_expressions)
                    if node.name == 'sum':
                        return add_expression
                    else:
                        # 对 mean 节点，需要除以 arity
                        return f"div({add_expression}, {node.arity})"
            else:
                # 其他节点正常处理
                return f"{node.name}({', '.join(child_expressions)})"
            
    def _generate_add_expression(self, child_expressions):
        """生成多个子节点相加的表达式"""
        if len(child_expressions) == 1:
            return child_expressions[0]
        else:
            return f"add({child_expressions[0]}, {self._generate_add_expression(child_expressions[1:])})"
        
    
    def preorder_to_latex(self,expression,fid):
        tokens = expression.replace("(", " ( ").replace(")", " ) ").replace(",", " ").split()
        return f"f_{{{fid}}}(X) = " + self._parse_expression(tokens)
    
    def _parse_expression(self,tokens):
        if not tokens:
            return ""
        
        token = tokens.pop(0)
        
        # 如果是函数节点
        if token in ["sum", "mean", "sqrt", "log", "neg", "abs", "sin", "cos", "tanh", "exp", "add", "sub", "mul", "div", "pow"]:
            arity = self.get_arity(token)
            args = []
            for _ in range(arity):
                # 跳过 "(" 和 ","
                while tokens and tokens[0] in ["(", ","]:
                    tokens.pop(0)
                args.append(self._parse_expression(tokens))
            # 跳过 ")"
            while tokens and tokens[0] == ")":
                tokens.pop(0)
            
            # 处理聚合函数
            if token in ["sum", "mean"]:
                return self._handle_aggregation(token, args)
            # 处理其他函数
            return self._handle_function(token, args)
        else:
            # 处理变量或常数
            return self._handle_variable(token)

    def _handle_function(self,func, args):
        if func == "add":
            return f"{args[0]} + {args[1]}"
        elif func == "sub":
            return f"{args[0]} - {args[1]}"
        elif func == "mul":
            return f"{args[0]} \\times {args[1]}"
        elif func == "div":
            return f"\\frac{{{args[0]}}}{{{args[1]}}}"
        elif func == "pow":
            return f"({args[0]})^{{{args[1]}}}"
        elif func == "neg":
            return f"-({args[0]})"
        elif func == "sqrt":
            return f"\\sqrt{{{args[0]}}}"
        elif func == "log":
            return f"\\log{{{args[0]}}}"
        elif func == "exp":
            return f"e^{{{args[0]}}}"
        elif func == "abs":
            return f"\\vert {{{args[0]}}} \\vert"
        # elif func == "pow":
        #     return f"{{{args[0]}}}^{{{args[1]}}}"
        elif func in ["sin", "cos", "tanh"]:
            return f"\\{func}({args[0]})"
        else:
            return func

    def _handle_variable(self,token):
        if token == "X":
            return "X_i"
        elif token == "X1":
            return "X_{i}"
        elif token == "X2":
            return "X_{i+1}"
        else:
            # 常数形式
            # 也标明下标位置
            if token[0] == "C":
                return f"{token}" + "_{i}"
            else:
                # mean的平均
                return f"{token}"

    def _handle_aggregation(self,func, args):
        # 检查子节点中是否包含 X, X1, X2
        # print(func,args)
        has_X = any("X_i" in arg for arg in args)
        has_X1 = any("X_{i}" in arg for arg in args)
        has_X2 = any("X_{i+1}" in arg for arg in args)
        
        # 确定聚合的上下标
        if has_X:
            lower = "i=0"
            upper = "n-1"
            divisor = "n" if func == "mean" else ""
        elif has_X1 or has_X2:
            lower = "i=0"
            upper = "n-2"
            divisor = "n-1" if func == "mean" else ""
        else:
            lower = "i=0"
            upper = "n-1"
            divisor = "n" if func == "mean" else ""
        
        # 生成 LaTeX 代码
        if func == "sum":
            if len(args) == 1:
                return f"\\sum_{{{lower}}}^{{{upper}}} ({args[0]})"
            else:
                return f"\\sum_{{{lower}}}^{{{upper}}} ({args[0]})"
        elif func == "mean":
            if len(args) == 1:
                return f"\\frac{{1}}{{{divisor}}} \\sum_{{{lower}}}^{{{upper}}} ({args[0]})"
            else: 
                return f"\\frac{{1}}{{{divisor}}} \\sum_{{{lower}}}^{{{upper}}} ({args[0]})"

    def get_arity(self,func):
        if func in ["sum", "mean", "sqrt", "log", "neg", "abs", "sin", "cos", "tanh", "exp"]:
            return 1
        elif func in ["add", "sub", "mul", "div", "pow"]:
            return 2
        else:
            return 0



    depth_ = property(_depth)
    length_ = property(_length)
    # indices_ = property(_indices)


    def replace_functions(self):
        """将 program 中的旧 _Function 对象替换为新的 _Function 对象"""
        if self.program is None:
            return

        new_program = []
        for item in self.program:
            if isinstance(item, tuple):  # 变量向量
                new_program.append(item)
            elif isinstance(item, list):  # 常数向量
                new_program.append(item)
            else:
                # 创建新的 _Function 对象，复制旧的成员变量
                # print(item.function)
                                # 根据 item.name 选择新的函数
                function_map = {
                    'sum': _sum,
                    'prod': _prod,
                    'mean': _mean,
                    'sqrt': _protected_sqrt,
                    'log': _protected_log,
                    'neg': np.negative,
                    'abs': np.abs,
                    'sin': np.sin,
                    'cos': np.cos,
                    'tan': np.tan,
                    'tanh': _tanh,
                    'exp': _protected_exp,
                    'inv': _protected_inverse,
                    'sig': _sigmoid,
                    'add': np.add,
                    'sub': _protected_sub,
                    'mul': np.multiply,
                    'div': _protected_division,
                    'max': np.maximum,
                    'min': np.minimum,
                    'pow': _protected_power
                    }
                if item.name not in function_map:
                    raise ValueError(f"Unsupported function name: {item.name}")
                new_function = _Function(
                    function=function_map[item.name],
                    name=item.name,
                    arity=item.arity,
                    input_dimension=item.input_dimension,
                    output_dimension=item.output_dimension,
                    depth=item.depth
                )
                # 复制其他成员变量
                new_function.remaining = item.remaining
                new_function.total = item.total
                new_function.constant_num = item.constant_num
                new_function.value_range = item.value_range
                new_function.parent_distance = item.parent_distance
                new_function.child_distance_list = item.child_distance_list
                new_program.append(new_function)

        self.program = new_program
