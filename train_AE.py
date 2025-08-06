import os
import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.transforms import Normalize
from net.AE import Autoencoder, load_data, train_autoencoder,make_dataset,split_data
import numpy as np
# from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
# from torch.utils.tensorboard import SummaryWriter
from dataset.bbob import *
from pflacco.sampling import create_initial_sample
from ela_feature import get_ela_feature
import pickle
from utils import set_seed


save_bbob_ela = './ela_store'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集生成函数
def generate_dataset(instance_num,suit='bbob',dim=10,n_fea = 21 , checkpoint = 15):
    # 初始化数据集
    dataset = np.zeros((24, instance_num, n_fea ))
    # lhs采样 Xs
    Xs = np.array(create_initial_sample(dim,n = 250 * dim,sample_type = 'lhs',lower_bound=-5,upper_bound=5,seed = 100))
    time_cost = 0
    for instance_idx in range(1,instance_num + 1):
        # 从 BBOB 数据集获取训练集和测试集
        # 设置不同的seed来获取不同的实例
        train_set, test_set = BBOB_Dataset.get_datasets(suit=suit, dim=dim,\
                                                        shifted=True,rotated=True,biased=False,\
                                                        instance_seed = instance_idx)
        # 遍历 24 个问题类
        final_time = 0
        for problem_idx, problem in enumerate(train_set.data + test_set.data):
            # Xs = -5 + 10 * np.random.rand(n_samples, dim)
            Ys = problem.eval(Xs)
            # 计算 ELA 特征
            features, _, total_calculation_time_cost = get_ela_feature(problem, Xs, Ys,random_state=100)
            #累计时间
            time_cost += total_calculation_time_cost
            # 保存到数据集中
            dataset[problem_idx, instance_idx - 1, :] = features[:n_fea]
            final_time = time_cost - final_time
        print(f'24 problems instance in {instance_idx} round : ',final_time)
        #get 15 rounds, save the data
        if instance_idx % checkpoint == 0:
            with open(os.path.join(save_bbob_ela,f'{dim}D_BBOB_ela_{checkpoint}_{instance_idx}.pickle'), 'wb') as f:
                pickle.dump(dataset, f, 0)
    print(f'total cost 24 * {instance_num} instance : ' ,time_cost)
    with open(os.path.join(save_bbob_ela,f'{dim}D_BBOB_ela_{n_fea}fea.pickle'), 'wb') as f:
        pickle.dump(dataset, f, 0)
    return dataset


def normalize_data(data):
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, data.shape[-1]))
    return data_normalized, scaler

if __name__ == '__main__':
    # train_AE()
    dim = 50
    batch_size = 32
    num_epochs = 300
    #24*270 instances per dimension
    instance_num = 270
    set_seed(100)
    dataset = generate_dataset(instance_num,dim=dim,n_fea=21)
    # with open(os.path.join(save_bbob_ela,f'{dim}D_BBOB_ela_21fea.pickle'), 'rb') as f:
        # dataset = pickle.load(f)
    n_fea = dataset.shape[-1] 
    train_data,val_data = split_data(dataset,val_split=0.2,random_state=42)
    #minmax特征归一化
    normalized_train_data, scaler = normalize_data(train_data)
    normalized_test_data = scaler.transform(val_data)
    train_loader,val_loader = make_dataset(normalized_train_data,normalized_test_data,batch_size)

    # Initialize model
    model = Autoencoder(n_fea)
    # Train the model
    train_autoencoder(model, train_loader, val_loader, lr=1e-3,num_epochs=num_epochs)

    # Save the final model
    torch.save(model.state_dict(), f"final_autoencoder_{n_fea}fea.pth")
    # print("Final model saved as final_autoencoder.pth")
