from dataset.bbob import Basic_Problem
from gplearn_final_AE_ray_best_ela.functions import _Function
from gplearn_final_AE_ray_best_ela.functions import _protected_division,_protected_exp,_protected_log,\
    _protected_power ,_protected_sqrt\
        ,_sum,_mean,_tanh
import numpy as np



class GP_problem(Basic_Problem):
    def __init__(self,execute,problemID,lb,ub,dim,random_state):
        self.problem = execute
        self.lb = lb
        self.ub = ub
        self.optimum = None
        self.problemID = problemID
        self.dim = dim
        self.random_state = random_state
        self.T1 = 0
        
    def func(self,x):
        return self.problem(x,self.random_state)
    
    def __call__(self, x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x)
    
    def __str__(self):
        return f'GP_Problem_{self.problemID}'
    
    def __name__(self):
        return f'GP_Problem_{self.problemID}'
    
class GP_problem_eval(Basic_Problem):
    def __init__(self,expr,eval_expr,constants,variables,problemID,lb,ub,dim,random_state):
        self.eval_expr = eval_expr
        self.expr = expr
        self.constants = constants
        self.locals_dict = {**constants,**variables}
        self.variables = variables
        # 在初始化时编译目标表达式
        # self._compiled_expr = compile(self.eval_expr, "<string>", "eval")
        self.lb = lb
        self.ub = ub
        self.optimum = None
        self.problemID = problemID
        self.dim = dim
        self.random_state = random_state
        
    def func(self,x):
        # 传入前先进行编译
        # 将 x 添加到 locals_dict 中
        self.locals_dict['x'] = x
        # 解析 Variables 中的表达式
        for key, value in self.variables.items():
            # print(self.locals_dict[key])
            self.locals_dict[key] = eval(value, globals(), self.locals_dict)
            # print("variable : " ,self.locals_dict[key].shape)
        for key,value in self.constants.items():
            # 调整常数向量的shape，进行样本数维度上的repeat
            if value.shape[-1] != 1:
                # 除了维度数为1的常数向量无需添加样本维度，其他的都需要添加样本维度，防止聚合函数出问题
                self.locals_dict[key] = np.repeat([value],x.shape[0],axis=0)
            # print("constant : " , self.locals_dict[key].shape)
        # 计算目标表达式
        # return eval(self._compiled_expr, globals(), self.locals_dict)
        # print(self.eval_expr)
        # print(self.locals_dict)
        return eval(self.eval_expr, globals(), self.locals_dict)

    
    def __call__(self, x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x)
    
    def __str__(self):
        return f'GP_Problem_{self.problemID}'
    
    def __name__(self):
        return f'GP_Problem_{self.problemID}'
    
    def get_latex(self):
        return preorder_to_latex(self.expr,self.problemID + 1)
    
    def get_latex_with_constants(self):
        expression  = preorder_to_latex(self.expr,self.problemID + 1)
        # 生成常数向量的解释
        constants_explanation = []
        for const_name, const_value in self.constants.items():
            if const_value.shape[-1] != 1:
                const_value_str = np.array2string(const_value, separator=', ')
                constants_explanation.append(f"{const_name} = {const_value_str}")

        # 将常数解释拼接为一行小字
        constants_explanation_str = "\\noindent \\quad \\text{where : } \n \\begin{quote} \n \(" + "\) \\\\ \( ".join(constants_explanation) + '\)' + "\n \\end{quote}"

        # 返回完整的 LaTeX 字符串
        return '\\begin{dmath*} \n' + f"{expression}" + '\n \\end{dmath*} \n' +  f"{constants_explanation_str}"
    
def preorder_to_latex(expression,fid):
    tokens = expression.replace("(", " ( ").replace(")", " ) ").replace(",", " ").split()
    return f"f_{fid}(X) = " + _parse_expression(tokens)

def _parse_expression(tokens):
    if not tokens:
        return ""
    
    token = tokens.pop(0)
    
    # 如果是函数节点
    if token in ["sum", "mean", "sqrt", "log", "neg", "abs", "sin", "cos", "tanh", "exp", "add", "sub", "mul", "div", "pow"]:
        arity = get_arity(token)
        args = []
        for _ in range(arity):
            # 跳过 "(" 和 ","
            while tokens and tokens[0] in ["(", ","]:
                tokens.pop(0)
            args.append(_parse_expression(tokens))
        # 跳过 ")"
        while tokens and tokens[0] == ")":
            tokens.pop(0)
        
        # 处理聚合函数
        if token in ["sum", "mean"]:
            return _handle_aggregation(token, args)
        # 处理其他函数
        return _handle_function(token, args)
    else:
        # 处理变量或常数
        return _handle_variable(token)

def _handle_function(func, args):
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
        return f"-{args[0]}"
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

def _handle_variable(token):
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

def _handle_aggregation(func, args):
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
            return f"\\sum_{{{lower}}}^{{{upper}}} {args[0]}"
        else:
            return f"\\sum_{{{lower}}}^{{{upper}}} ({args[0]})"
    elif func == "mean":
        if len(args) == 1:
            return f"\\frac{{1}}{{{divisor}}} \\sum_{{{lower}}}^{{{upper}}} {args[0]}"
        else: 
            return f"\\frac{{1}}{{{divisor}}} \\sum_{{{lower}}}^{{{upper}}} ({args[0]})"

def get_arity(func):
    if func in ["sum", "mean", "sqrt", "log", "neg", "abs", "sin", "cos", "tanh", "exp"]:
        return 1
    elif func in ["add", "sub", "mul", "div", "pow"]:
        return 2
    else:
        return 0