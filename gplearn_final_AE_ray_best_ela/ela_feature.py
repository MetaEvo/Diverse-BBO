from pflacco.classical_ela_features \
import calculate_ela_meta, calculate_ela_conv, calculate_ela_distribution,calculate_information_content
import math
import numpy as np

def get_ela_feature(problem, Xs, Ys, random_state):
    total_calculation_time_cost = 0
    total_calculation_fes = 0
    all_features = []
    
    # 计算下列特征时 对Y进行归一化
    # 为了解决problem.eval无法归一化
    # 传入原始的Y来进行归一化操作
    # convexity features 4 features
    ela_conv_full_results = calculate_ela_conv(Xs, Ys, problem.eval, ela_conv_nsample= 200 , seed=random_state)
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
        # elif k == 'ela_conv.costs_runtime':
            # print("conv feature costs : ",ela_conv_full_results[k])
        
        
    # 计算下列特征时 对Y进行归一化
    # 对目标值进行归一化
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
        # else:
            # print("ic feature costs : ",ela_ic_full_results[k])
        
    # print("check conv ",all_features)


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
        # else:
            # print("dist feature costs : ",ela_dis_full_results[k])
    # print(all_features)

    return np.array(all_features), total_calculation_fes, total_calculation_time_cost
