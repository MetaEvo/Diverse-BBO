import os 
import pickle
from dataset.GP import GP_problem
import numpy as np
def get_gp_function(upperbound = 5,func_dir ='./save_gp_functions' ,random_state = 42):
    train_func = []
    with open(os.path.join(func_dir,f'all_256_programs.pickle'),'rb') as f:
        programs = pickle.load(f)  
        f.close()
    for program in programs:
        train_func.append(GP_problem(program.execute,program.problemID,lb=-upperbound,ub=upperbound,dim = program.best_dim , random_state=random_state ))
    return train_func

func_dir = './save_gp_functions'
function_list = get_gp_function(upperbound =5,func_dir = func_dir) #256 functions

for function in function_list[:2]:
    # eval function in its best dim
    # x is bounded to [-5.,5.]
    # use .func() to evaluate the function
    np.random.seed(42)
    x = np.random.rand(10000,function.dim)
    y = function.func(x)
    print(f"Function {function.problemID} : min : {np.min(y)} , max : {np.max(y)}")
    
