from pflacco_v1.classical_ela_features \
import calculate_ela_meta, calculate_ela_conv,calculate_ela_distribution,\
        calculate_information_content
import math
import numpy as np

def get_ela_feature(problem, Xs, Ys,random_state):
    total_calculation_time_cost = 0
    total_calculation_fes = 0
    all_features = []
    
    all_ela_keys = []
    
    # convexity features 4 features
    ela_conv_full_results = calculate_ela_conv(Xs, Ys, problem, ela_conv_nsample= 200,seed=random_state)
    total_calculation_fes += ela_conv_full_results['ela_conv.additional_function_eval']
    total_calculation_time_cost += ela_conv_full_results['ela_conv.costs_runtime']
    # print('convexity features time cost: {},  consumed fes: {}'.format(ela_conv_full_results['ela_conv.costs_runtime'], ela_conv_full_results['ela_conv.additional_function_eval']))
    for k in ela_conv_full_results.keys():
        if (k != 'ela_conv.additional_function_eval') and (k != 'ela_conv.costs_runtime'):
            v = ela_conv_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # elif k == 'ela_conv.costs_runtime':
            # print("conv feature costs : ",ela_conv_full_results[k])
        
        
    Ys = (Ys - Ys.min()) / (Ys.max() - Ys.min())
    
    # meta features 9 features
    ela_meta_full_results = calculate_ela_meta(Xs,Ys)
    total_calculation_time_cost += ela_meta_full_results['ela_meta.costs_runtime']
    # print('meta features time cost: {},  consumed fes: {}'.format(ela_meta_full_results['ela_meta.costs_runtime'], 0))
    for k in ela_meta_full_results.keys():
        if k != 'ela_meta.costs_runtime':
            v = ela_meta_full_results[k]
            # print(f"{k}: {v}")
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # else:
            # print("meta feature costs : ",ela_meta_full_results[k])
    # print("check meta" ,len(all_features))
    
    #information content 5 features 
    ela_ic_full_results = calculate_information_content(Xs,Ys,seed=random_state)
    total_calculation_time_cost += ela_ic_full_results['ic.costs_runtime']
    # print('ic features time cost: {},  consumed fes: {}'.format(ela_ic_full_results['ic.costs_runtime'], 0))
    for k in ela_ic_full_results.keys():
        if k != 'ic.costs_runtime':
            v = ela_ic_full_results[k]
            # print(f"{k}: {v}")
            if v is None:
                v = 0.
            elif math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # else:
            # print("ic feature costs : ",ela_ic_full_results[k])
        
    # print("check conv ",all_features)
    # level-set features
    # try:
    #     ela_level_full_results = calculate_ela_level(Xs,Ys, ela_level_quantiles=[0.25, 0.5, 0.75], ela_level_resample_iterations=10)
    # except:
    #     ela_level_full_results = {}
    #     for _k in range(9):
    #         ela_level_full_results[_k] = -1
    #     ela_level_full_results['ela_level.costs_runtime'] = 0
    # total_calculation_time_cost += ela_level_full_results['ela_level.costs_runtime']
    # #print('level-set features time cost: {},  consumed fes: {}'.format(ela_level_full_results['ela_level.costs_runtime'], 0))
    # for k in ela_level_full_results.keys():
    #     if k != 'ela_level.costs_runtime':
    #         v = ela_level_full_results[k]
    #         if math.isnan(v):
    #             v = 0.
    #         elif math.isinf(v):
    #             v = 1.
    #         all_features.append(v)


    # curvature features
    # ela_cur_full_results = calculate_ela_curvate(Xs, Ys, problem.eval, problem.dim, problem.lb, problem.ub, sample_size_factor = 1)
    # total_calculation_fes += ela_cur_full_results['ela_curv.costs_fun_evals:']
    # total_calculation_time_cost += ela_cur_full_results['ela_curv.costs_runtime']
    # print('curvature features time cost: {},  consumed fes: {}'.format(ela_cur_full_results['ela_curv.costs_runtime'], ela_cur_full_results['ela_curv.costs_fun_evals:']))
    # for k in ela_cur_full_results.keys():
    #     if (k != 'ela_curv.costs_fun_evals:') and (k != 'ela_curv.costs_runtime'):
    #         all_features.append(ela_cur_full_results[k])

    # local features
    # ela_local_full_results = calculate_ela_local(Xs, Ys, problem.eval, problem.dim, problem.lb, problem.ub,ela_local_local_searches_factor=1)
    # total_calculation_fes += ela_local_full_results['ela_local.additional_function_eval']
    # total_calculation_time_cost += ela_local_full_results['ela_local.costs_runtime']
    # #print('local features time cost: {},  consumed fes: {}'.format(ela_local_full_results['ela_local.costs_runtime'],
    #                                                               #ela_local_full_results['ela_local.additional_function_eval']))
    # for i,k in enumerate(ela_local_full_results.keys()):
    #     if i <= 6:
    #         v = ela_local_full_results[k]
    #         if math.isnan(v):
    #             v = 0.
    #         elif math.isinf(v):
    #             v = 1.
    #         all_features.append(v)

    # distributional features 3 features
    ela_dis_full_results = calculate_ela_distribution(Xs,Ys)
    total_calculation_time_cost += ela_dis_full_results['ela_distr.costs_runtime']
    # print('distributional features time cost: {},  consumed fes: {}'.format(ela_dis_full_results['ela_distr.costs_runtime'], 0))
    for k in ela_dis_full_results.keys():
        if k != 'ela_distr.costs_runtime':
            v = ela_dis_full_results[k]
            if math.isnan(v):
                v = 0.
            elif math.isinf(v):
                v = 1.
            all_features.append(v)
            all_ela_keys.append(k)
        # else:
            # print("dist feature costs : ",ela_dis_full_results[k])
    # print(all_features)
    # all_features = [fea for j, fea in enumerate(all_features) if j not in [1,2,3,16,17]]
    return np.array(all_features), total_calculation_fes, total_calculation_time_cost



if __name__ == '__main__':
    from dataset.bbob import *
    train_set, test_set = BBOB_Dataset.get_datasets(suit='bbob',
                                                    dim=10,
                                                    upperbound=5,
                                                    shifted=False,
                                                    rotated=False,
                                                    biased=False)
    p = train_set[0]
    np.random.seed(1)
    Xs = -5 + 10 * np.random.rand(1000,10)
    Ys = p.eval(Xs)
    features = get_ela_feature(p,Xs,Ys)
    print(features)


