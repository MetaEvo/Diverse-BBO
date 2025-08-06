
# from gplearn_no_restriction.genetic import SymbolicRegressor
from sklearn.preprocessing import MinMaxScaler
# from gplearn_final_AE_ray_ver2_no_closure.genetic import SymbolicRegressor
from gplearn_final_AE_ray_best_ela.genetic import SymbolicRegressor
import os 
import pickle
import numpy as np
from pflacco.sampling import create_initial_sample
from net.AE import *
from datetime import datetime
import ray
import copy
import argparse

class benchmark_generator():
    def __init__(self,dim,seed,model_path,n_fea = 21) :
        self.random_state = seed
        self.dim = dim
        self.X = np.array(create_initial_sample(dim, n=250*dim, sample_type='lhs', lower_bound=-5, upper_bound=5, seed=seed))
        self.model = load_model(model_path,n_fea=n_fea)
        with open(os.path.join('./scaler_data/scaler.pickle'), 'rb') as f:
            scaler = pickle.load(f)
        self.scaler = scaler
        sample_points_path = './sample_points/2024_10_16_160100_final_samplepoints'
        self.sample_problems = np.load(os.path.join(sample_points_path,f'sample_points_10D.npy')).reshape(-1,2)
        # nowTime=datetime.now().strftime('%Y_%m_%d_%H%M%S')
        # self.save_path = './save_gp_functions/'+nowTime
        self.save_path = './save_gp_functions/new_functions'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

def recover_sample(bench,problemID):
    y = np.zeros(bench.X.shape[0])
    print(f"----------------func{problemID}---------------------")
    est_gp = SymbolicRegressor( population_size=1000,
                            generations=50, stopping_criteria=0.005,
                            p_crossover=0.6, p_subtree_mutation=0.25,
                            p_hoist_mutation=0.04, p_point_mutation=0.1,
                            parsimony_coefficient=0.001,
                            tournament_size=50,
                            random_state=bench.random_state + problemID + 128,
                            problemID=problemID,
                            problem_coord=bench.sample_problems[problemID-1],
                            n_jobs=10,
                            metric='mse',
                            init_depth=(5,8),
                            mutate_depth=(5,15),
                            variable_range=(-5.0,5.0),
                            init_method='full',
                            model=bench.model,
                            dim = bench.dim,
                            scaler=bench.scaler,
                            function_set=['add','sub','mul','div','sum','pow',
                                            'mean','sin','cos','tanh','log','sqrt','abs','neg','exp'],
                            save_path = bench.save_path)
    function_info = est_gp.fit(bench.X,y)
    return function_info



if __name__ == '__main__':
    ray.init()

    
    # # 初始化 argparse
    parser = argparse.ArgumentParser(description="Run benchmark generator with specified problem range.")
    parser.add_argument('--func_id', type=int, required=True, help="The starting problemID for this batch.")
    # parser.add_argument('--end_id', type=int, required=True, help="The ending problemID for this batch.")
    parser.add_argument('--dim', type=int, default=10, help="The dimension.")
    parser.add_argument('--seed', type=int, default=100, help="The random seed.")
    args = parser.parse_args()

    # 从命令行参数读取 problemID 范围
    func_id = args.func_id
    seed = args.seed
    dim = args.dim

    model_path = './models/autoencoder_epoch_300.pth'
    benchmark_gen = benchmark_generator(dim, seed, model_path)
    info_dict = recover_sample(copy.deepcopy(benchmark_gen),func_id)
    # for info_dict in results:
    for info_store in list(info_dict.keys()):
        # 根据keys的后缀分别存放不同类型的文件
        if info_store.split('.')[-1] == 'pickle':
            with open(os.path.join(benchmark_gen.save_path,info_store),'wb')as f1:
                pickle.dump(info_dict[info_store],f1,0)
                f1.close()
        else:
            with open(os.path.join(benchmark_gen.save_path,info_store),'w')as f2:
                for info_txt in info_dict[info_store]:
                    f2.write(info_txt)
                f2.close()
    
    