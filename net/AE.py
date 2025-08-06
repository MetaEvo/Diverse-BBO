import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
from sklearn.model_selection import train_test_split



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
class Autoencoder(nn.Module):
    def __init__(self,n_fea):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_fea, 128),  # 增加神经元数量
            nn.PReLU(),
            # nn.Linear(128, 256),  # 增加神经元数量
            # nn.PReLU(),
            nn.Linear(128,64),  # 增加神经元数量
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, 8),
            nn.PReLU(),
            nn.Linear(8, 2),  # 最终降维到2维
            nn.Tanh()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.PReLU(),
            nn.Linear(8, 16),
            nn.PReLU(),
            nn.Linear(16, 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64,128),
            nn.PReLU(),
            # nn.Linear(64,256),
            # nn.PReLU(),
            # nn.Linear(256,128),
            # nn.PReLU(),
            nn.Linear(128, n_fea)  # 输出层
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.encoder(x)
        # x = 5 * x 
        x = self.decoder(x)
        return x

def train_autoencoder(model, train_loader, val_loader, num_epochs=100, save_interval=50,lr = 1e-4, model_dir="./models", log_dir="./logs"):
    nowTime=datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    # model_dir = os.path.join(model_dir,nowTime)
    # log_dir = os.path.join(log_dir,nowTime)
    model_dir = model_dir+'/'+nowTime
    log_dir = log_dir+'/'+nowTime
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model.to(device)
    # writer = SummaryWriter(log_dir)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.9862327,last_epoch=-1)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # writer.add_scalars('Loss', {'train':avg_train_loss}, epoch)
        # writer.add_scalars('Loss', {'val':avg_val_loss}, epoch)
        # writer.add_scalar('learnrate_pg', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        scheduler.step()
        
        if (epoch + 1) % save_interval == 0:
            model_path = os.path.join(model_dir,f"autoencoder_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f'Model saved to {model_path}')

    # writer.close()

def load_data(data, batch_size=16, val_split=0.2):
    data = data.reshape(-1, data.shape[-1])  # Flatten to shape (24*1280, n_fea)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    # return train_dataset, val_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
def split_data(data,val_split=0.2,random_state = 42):
    data = data.reshape(-1, data.shape[-1])
    train_data, val_data = train_test_split(data, test_size=val_split, random_state=random_state,shuffle=True)
    return train_data,val_data

def make_dataset(train_data,val_data, batch_size=16):
    train_data,val_data = train_data.reshape(-1, train_data.shape[-1]) ,val_data.reshape(-1,val_data.shape[-1]) # Flatten to shape (24*1280, n_fea)
    train_data,val_data = TensorDataset(torch.tensor(train_data, dtype=torch.float32)),TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_model(model_path,n_fea):
    '''
    model_path:the path of save model's args,
    n_fea: 32(or the number of feature)
    '''
    # 加载模型
    model = Autoencoder(n_fea).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def get_encoded(model,input_features):
    '''
    model:AE model,
    input_feature:last dim must be 32(or the number of feature)
    '''
    model.eval()
    # 将输入特征转换为张量
    input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    # 仅使用编码器部分进行降维
    with torch.no_grad():
        #放缩回原来的位置
        encoded_features = model.encoder(input_tensor).cpu().numpy() * 5
    return encoded_features



if __name__ == "__main__":
    n_fea = 128  # Example input feature size (replace with actual n_fea)
    batch_size = 32
    num_epochs = 100

    # Simulate loading your actual data with shape (24, 64, n_fea)
    data = np.random.rand(24, 64, n_fea)

    # Load data
    train_loader, val_loader = load_data(data, batch_size=batch_size)

    # Initialize model
    model = Autoencoder(n_fea)

    # Train the model
    train_autoencoder(model, train_loader, val_loader, num_epochs=num_epochs)

    # Save the final model
    torch.save(model.state_dict(), "final_autoencoder.pth")
    print("Final model saved as final_autoencoder.pth")
