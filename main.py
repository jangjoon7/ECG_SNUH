#%%
import numpy as np
import os
import pandas
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from sklearn.linear_model import LinearRegression

###############################################################


dataAdultTrainFolder = './data_train_adult_npy/'
dataAdultValidFolder = './data_valid_adult_npy/'
dataChildTrainFolder = './data_train_child_npy/'
dataChildValidFolder = './data_valid_child_npy/'

# folder_path = pathlib.Path("/data_train/train/adult/age/")
# folder_path.mkdir(parents=True, exist_ok=True)

ageFolder = './data_train_age/'
ageFileName = 'ECG_adult_age_train.csv'
dataConcat = './data/'

####################################################################################

datalist = os.listdir(dataAdultTrainFolder)
NumData = len(os.listdir(dataAdultTrainFolder))

max_number = max([int(re.search(r'\d+', file_name).group()) for file_name in datalist])

X_data = np.zeros((NumData, 60000))
idx_tmp = 0

for i in range(max_number):
    if datalist.count('ecg_adult_' + str(i) + '.npy') == 1:
        X_data[idx_tmp, :] = np.load(dataAdultTrainFolder + 'ecg_adult_' + str(i) + '.npy')
        idx_tmp += 1

Y_data = np.array(pandas.read_csv(ageFolder + ageFileName)['AGE'])

np.savez_compressed(dataConcat + 'Adult_train', X_data=X_data, Y_data=Y_data)

datalist = os.listdir(dataAdultValidFolder)
NumData = len(os.listdir(dataAdultValidFolder))

max_number = max([int(re.search(r'\d+', file_name).group()) for file_name in datalist])

X_data = np.zeros((NumData, 60000))
idx_tmp = 0

for i in range(max_number):
    if datalist.count('ecg_adult_' + str(i) + '.npy') == 1:
        X_data[idx_tmp, :] = np.load(dataAdultValidFolder + 'ecg_adult_' + str(i) + '.npy')
        idx_tmp += 1

np.savez_compressed(dataConcat + 'Adult_valid', X_data=X_data)

datalist = os.listdir(dataChildTrainFolder)
NumData = len(os.listdir(dataChildTrainFolder))

max_number = max([int(re.search(r'\d+', file_name).group()) for file_name in datalist])

X_data = np.zeros((NumData, 60000))
idx_tmp = 0

for i in range(max_number):
    if datalist.count('ecg_child_' + str(i) + '.npy') == 1:
        X_data[idx_tmp, :] = np.load(dataChildTrainFolder + 'ecg_child_' + str(i) + '.npy')
        idx_tmp += 1

Y_data = np.array(pandas.read_csv(ageFolder + ageFileName)['AGE'])

np.savez_compressed(dataConcat + 'Child_train', X_data=X_data, Y_data=Y_data)

datalist = os.listdir(dataChildValidFolder)
NumData = len(os.listdir(dataChildValidFolder))

max_number = max([int(re.search(r'\d+', file_name).group()) for file_name in datalist])

X_data = np.zeros((NumData, 60000))
idx_tmp = 0

for i in range(max_number):
    if datalist.count('ecg_child_' + str(i) + '.npy') == 1:
        X_data[idx_tmp, :] = np.load(dataChildValidFolder + 'ecg_child_' + str(i) + '.npy')
        idx_tmp += 1

np.savez_compressed(dataConcat + 'Child_valid', X_data=X_data)

data = torch.from_numpy(np.load(dataConcat + 'Adult_train.npz')['X_data'])
data = data.reshape(data.shape[0], 12, 5000)
data_f = np.zeros((data.shape[0], 12, 5000))

for i in range(data.shape[0]):
    for lead in range(12):
        temp = np.array(data[i, lead])
        p = np.polyfit(np.linspace(0, 10, 5000), np.array(data[i, lead]), 11)
        f = np.polyval(p, np.linspace(0, 10, 5000))
        data_f[i, lead] = temp - f
data_f = torch.from_numpy(data_f).to(torch.float)

data_f = data_f.reshape(data_f.shape[0], 12*5000)
data_f -= data_f.min(1)[0].unsqueeze(1)
data_f /= data_f.max(1)[0].unsqueeze(1)
data_f = data_f.reshape(data_f.shape[0], 12, 5000)

nans = np.unique(np.where(data_f.isnan())[0])
selected_data = torch.index_select(data_f, dim=0, index=torch.tensor([i for i in range(data_f.size(0)) if i not in nans]))

X_data = np.array(selected_data)
Y_data = torch.from_numpy(np.load(dataConcat + 'Adult_train.npz')['Y_data'])
Y_data = torch.index_select(Y_data, dim=0, index=torch.tensor([i for i in range(Y_data.size(0)) if i not in nans]))
Y_data = np.array(Y_data)

np.savez_compressed(dataConcat + 'Adult_train_f.npz', X_data=X_data, Y_data=Y_data)

data = torch.from_numpy(np.load(dataConcat + 'Adult_valid.npz')['X_data'])
data = data.reshape(data.shape[0], 12, 5000)
data_f = np.zeros((data.shape[0], 12, 5000))

for i in range(data.shape[0]):
    for lead in range(12):
        temp = np.array(data[i, lead])
        p = np.polyfit(np.linspace(0, 10, 5000), np.array(data[i, lead]), 11)
        f = np.polyval(p, np.linspace(0, 10, 5000))
        data_f[i, lead] = temp - f
data_f = torch.from_numpy(data_f).to(torch.float)

data_f = data_f.reshape(data_f.shape[0], 12*5000)
data_f -= data_f.min(1)[0].unsqueeze(1)
data_f /= data_f.max(1)[0].unsqueeze(1)
data_f = data_f.reshape(data_f.shape[0], 12, 5000)

nans = np.unique(np.where(data_f.isnan())[0])
#%%
total_adult_val = data_f.shape[0]
adult_valid_nans = nans.copy()
selected_data = torch.index_select(data_f, dim=0, index=torch.tensor([i for i in range(data_f.size(0)) if i not in nans]))

X_data = np.array(selected_data)

np.savez_compressed(dataConcat + 'Adult_valid_f.npz', X_data=X_data)

data = torch.from_numpy(np.load(dataConcat + 'Child_train.npz')['X_data'])
data = data.reshape(data.shape[0], 12, 5000)
data_f = np.zeros((data.shape[0], 12, 5000))

for i in range(data.shape[0]):
    for lead in range(12):
        temp = np.array(data[i, lead])
        p = np.polyfit(np.linspace(0, 10, 5000), np.array(data[i, lead]), 11)
        f = np.polyval(p, np.linspace(0, 10, 5000))
        data_f[i, lead] = temp - f
data_f = torch.from_numpy(data_f).to(torch.float)

data_f = data_f.reshape(data_f.shape[0], 12*5000)
data_f -= data_f.min(1)[0].unsqueeze(1)
data_f /= data_f.max(1)[0].unsqueeze(1)
data_f = data_f.reshape(data_f.shape[0], 12, 5000)

nans = np.unique(np.where(data_f.isnan())[0])
selected_data = torch.index_select(data_f, dim=0, index=torch.tensor([i for i in range(data_f.size(0)) if i not in nans]))

X_data = np.array(selected_data)
Y_data = torch.from_numpy(np.load(dataConcat + 'Child_train.npz')['Y_data'])
Y_data = torch.index_select(Y_data, dim=0, index=torch.tensor([i for i in range(Y_data.size(0)) if i not in nans]))
Y_data = np.array(Y_data)

np.savez_compressed(dataConcat + 'Child_train_f.npz', X_data=X_data, Y_data=Y_data)

data = torch.from_numpy(np.load(dataConcat + 'Child_valid.npz')['X_data'])
data = data.reshape(data.shape[0], 12, 5000)
data_f = np.zeros((data.shape[0], 12, 5000))

for i in range(data.shape[0]):
    for lead in range(12):
        temp = np.array(data[i, lead])
        p = np.polyfit(np.linspace(0, 10, 5000), np.array(data[i, lead]), 11)
        f = np.polyval(p, np.linspace(0, 10, 5000))
        data_f[i, lead] = temp - f
data_f = torch.from_numpy(data_f).to(torch.float)

data_f = data_f.reshape(data_f.shape[0], 12*5000)
data_f -= data_f.min(1)[0].unsqueeze(1)
data_f /= data_f.max(1)[0].unsqueeze(1)
data_f = data_f.reshape(data_f.shape[0], 12, 5000)

nans = np.unique(np.where(data_f.isnan())[0])
total_child_val = data_f.shape[0]
child_valid_nans = nans.copy()
selected_data = torch.index_select(data_f, dim=0, index=torch.tensor([i for i in range(data_f.size(0)) if i not in nans]))

X_data = np.array(selected_data)

np.savez_compressed(dataConcat + 'Child_valid_f.npz', X_data=X_data)

###############################################################

valnum = 2000

dat = np.load(dataConcat + 'Adult_train_f.npz')['X_data']
age = np.load(dataConcat + 'Adult_train_f.npz')['Y_data']
ageGaussian = 1/(1*np.sqrt(2*np.pi)) * np.exp((-1/2)*((np.expand_dims(np.linspace(19.5, 121.5, 103), 0)-np.expand_dims(np.array(age), 1))**2))

dat_half =  np.concatenate((dat[:-valnum, :, :2048], dat[:-valnum, :, 2048:4096]), 0)
dat_half_val =  np.concatenate((dat[-valnum:, :, :2048], dat[-valnum:, :, 2048:4096]), 0)

age_half =  np.concatenate((age[:-valnum], age[:-valnum]), 0)
age_half_val =  np.concatenate((age[-valnum:], age[-valnum:]), 0)

ageGaussian_half =  np.concatenate((ageGaussian[:-valnum], ageGaussian[:-valnum]), 0)

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/adult/dat/adult_train_'+ str(int(i)).zfill(5), dat_half[i])

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/adult/age/adult_train_'+ str(int(i)).zfill(5), age_half[i])

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/adult/ageGaussian/adult_train_'+ str(int(i)).zfill(5), ageGaussian_half[i])

for i in range(dat_half_val.shape[0]):
    np.save('./data_train/valid/adult/dat/adult_train_valid_'+ str(int(i)).zfill(5), dat_half_val[i])

for i in range(dat_half_val.shape[0]):
    np.save('./data_train/valid/adult/age/adult_train_valid_'+ str(int(i)).zfill(5), age_half_val[i])

dat = np.load(dataConcat + 'Child_train_f.npz')['X_data']
age = np.load(dataConcat + 'Child_train_f.npz')['Y_data']
ageGaussian = 1/(1*np.sqrt(2*np.pi)) * np.exp((-1/2)*((np.expand_dims(np.linspace(0.5, 18.5, 19), 0)-np.expand_dims(np.array(age), 1))**2))

dat_half =  np.concatenate((dat[:-valnum, :, :2048], dat[:-valnum, :, 2048:4096]), 0)
dat_half_val =  np.concatenate((dat[-valnum:, :, :2048], dat[-valnum:, :, 2048:4096]), 0)

age_half =  np.concatenate((age[:-valnum], age[:-valnum]), 0)
age_half_val =  np.concatenate((age[-valnum:], age[-valnum:]), 0)

ageGaussian_half =  np.concatenate((ageGaussian[:-valnum], ageGaussian[:-valnum]), 0)

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/child/dat/child_train_'+ str(int(i)).zfill(5), dat_half[i])

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/child/age/child_train_'+ str(int(i)).zfill(5), age_half[i])

for i in range(dat_half.shape[0]):
    np.save('./data_train/train/child/ageGaussian/child_train_'+ str(int(i)).zfill(5), ageGaussian_half[i])

for i in range(dat_half_val.shape[0]):
    np.save('./data_train/valid/child/dat/child_train_valid_'+ str(int(i)).zfill(5), dat_half_val[i])

for i in range(dat_half_val.shape[0]):
    np.save('./data_train/valid/child/age/child_train_valid_'+ str(int(i)).zfill(5), age_half_val[i])

dat = np.load(dataConcat + 'Adult_valid_f.npz')['X_data']
dat_half =  np.concatenate((dat[:, :, :2048], dat[:, :, 2048:4096]), 0)

for i in range(dat_half.shape[0]):
    np.save('./data_valid/adult/adult_valid_'+ str(int(i)).zfill(5), dat_half[i])

dat = np.load(dataConcat + 'Child_valid_f.npz')['X_data']
dat_half =  np.concatenate((dat[:, :, :2048], dat[:, :, 2048:4096]), 0)

for i in range(dat_half.shape[0]):
    np.save('./data_valid/child/child_valid_'+ str(int(i)).zfill(5), dat_half[i])

##############################################################################################################################

class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        
        dat = np.load(os.path.join(self.dat_dir, i_name))
        ageGaussian = np.load(os.path.join(self.root_dir, 'ageGaussian', i_name)).astype(np.float32)

        return {'dat':dat, 'ageGaussian':ageGaussian}
    
class ValidDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        
        dat = np.load(os.path.join(self.dat_dir, i_name))
        age = np.load(os.path.join(self.root_dir, 'age', i_name)).astype(np.float32)

        return {'dat':dat, 'age':age}

train_dataset = TrainDataset(root_dir='./data_train/train/adult/')
loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = ValidDataset(root_dir='./data_train/valid/adult/')
loader_valid = DataLoader(valid_dataset, batch_size=64, shuffle=False)

class SFCN(torch.nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.Convblock = nn.Sequential( nn.Conv1d(12, 32, 19, 1, 9),
                                        nn.BatchNorm1d(32),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(32, 64, 19, 1, 9),
                                        nn.BatchNorm1d(64),                        
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                    
                                        nn.Conv1d(64, 128, 19, 1, 9),
                                        nn.BatchNorm1d(128),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(128, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(256, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                        )
        
        self.Convblock2 = nn.Sequential(nn.Conv1d(256, 128, 1, 1, 0),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        )
        
        self.Ave = nn.AdaptiveAvgPool1d(1)

        self.Rest = nn.Sequential(  nn.Dropout(),
                                    nn.Conv1d(128, 103, 1, 1, 0),
                                    )
                                    
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Convblock(x)
        out = self.Convblock2(out)
        out = self.Ave(out)
        out = self.Rest(out)
        out = out.squeeze()
        out = self.Softmax(out)
        return out
    
for ensemble in range(20):

    model = nn.DataParallel(SFCN().to('cuda'))

    LossFnc = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    MAE1tmp = 7.5
    MAE2tmp = 7.8

    for epoch in range(20):

        model.train()
        
        for _, x in enumerate(loader_train):
            inp = x['dat'].to('cuda')
            Y = x['ageGaussian'].to('cuda')
            optimizer.zero_grad()
            predict = model(inp)
            
            cost = LossFnc(torch.log(predict + 1e-6), Y)
            cost.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            total_AE1 = torch.empty(0)
            total_AE2 = torch.empty(0)

            for _, x_val in enumerate(loader_valid):
                inp_val = x_val['dat'].to('cuda')
                Y_val = x_val['age'].to('cuda')
                valpredict = (model(inp_val)*torch.linspace(19.5, 121.5, 103).to('cuda')).sum(1)
                valAE1 = (Y_val - valpredict).abs()
                valAE2 = (Y_val - (19.5 + (model(inp_val).max(1)[1]))).abs()
                
                total_AE1 = torch.cat((total_AE1, valAE1.detach().cpu()))
                total_AE2 = torch.cat((total_AE2, valAE2.detach().cpu()))

        valMAE1 = (total_AE1[:total_AE1.shape[0]//2] + total_AE1[total_AE1.shape[0]//2:])/2
        valMAE2 = (total_AE2[:total_AE2.shape[0]//2] + total_AE2[total_AE2.shape[0]//2:])/2

        valMAE1 = valMAE1.mean()
        valMAE2 = valMAE2.mean()

        print('[Epoch: {:>4}] cost = {:>.3}, MAE1 = {:>.3}, MAE2 = {:>.3}'.format(epoch + 1, cost, valMAE1, valMAE2))

        if MAE1tmp > valMAE1:
            MAE1tmp = valMAE1
            if MAE2tmp > valMAE2:
                MAE2tmp = valMAE2
                torch.save(model.state_dict(), 'ENa' + str(ensemble).zfill(2) + '_model_MAE1_'+ str(valMAE1.item()) +'_MAE2_' + str(valMAE2.item()) + '_min.pth')
            else:
                torch.save(model.state_dict(), 'ENa' + str(ensemble).zfill(2) + '_model_MAE1_'+ str(valMAE1.item()) +'_MAE2_' + str(valMAE2.item()) + '.pth')

    model.eval()

files = os.listdir()
result_folder = "./ensemble_adult"

for ensemble in range(20):
    min_mae1 = float('inf')
    min_mae1_file = None

    for file in files:
        if file.startswith("ENa" + str(ensemble).zfill(2)) and file.endswith(".pth"):
            mae1_value = float(file.split("_")[3])
            if mae1_value < min_mae1:
                min_mae1 = mae1_value
                min_mae1_file = file
                print(min_mae1)

    if min_mae1_file:
        os.rename(min_mae1_file, os.path.join(result_folder, min_mae1_file))

    for file in files:
        if file.startswith("ENa" + str(ensemble).zfill(2)) and file.endswith(".pth"):
            if file != result_folder and os.path.isfile(file):
                os.remove(file)

class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        
        dat = np.load(os.path.join(self.dat_dir, i_name))
        ageGaussian = np.load(os.path.join(self.root_dir, 'ageGaussian', i_name)).astype(np.float32)

        return {'dat':dat, 'ageGaussian':ageGaussian}
    
class ValidDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        
        dat = np.load(os.path.join(self.dat_dir, i_name))
        age = np.load(os.path.join(self.root_dir, 'age', i_name)).astype(np.float32)

        return {'dat':dat, 'age':age}

train_dataset = TrainDataset(root_dir='./data_train/train/child/')
loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = ValidDataset(root_dir='./data_train/valid/child/')
loader_valid = DataLoader(valid_dataset, batch_size=64, shuffle=False)

class SFCN(torch.nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.Convblock = nn.Sequential( nn.Conv1d(12, 32, 19, 1, 9),
                                        nn.BatchNorm1d(32),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(32, 64, 19, 1, 9),
                                        nn.BatchNorm1d(64),                        
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                    
                                        nn.Conv1d(64, 128, 19, 1, 9),
                                        nn.BatchNorm1d(128),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(128, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(256, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                        )
        
        self.Convblock2 = nn.Sequential(nn.Conv1d(256, 64, 1, 1, 0),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        )
        
        self.Ave = nn.AdaptiveAvgPool1d(1)

        self.Rest = nn.Sequential(  nn.Dropout(),
                                    nn.Conv1d(64, 19, 1, 1, 0),
                                    )
                                    
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Convblock(x)
        out = self.Convblock2(out)
        out = self.Ave(out)
        out = self.Rest(out)
        out = out.squeeze()
        out = self.Softmax(out)
        return out
    
for ensemble in range(20):

    model = nn.DataParallel(SFCN().to('cuda'))

    LossFnc = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    MAE1tmp = 0.75
    MAE2tmp = 0.75

    for epoch in range(20):

        model.train()
        
        for _, x in enumerate(loader_train):
            inp = x['dat'].to('cuda')
            Y = x['ageGaussian'].to('cuda')
            optimizer.zero_grad()
            predict = model(inp)
            
            cost = LossFnc(torch.log(predict + 1e-6), Y)
            cost.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            total_AE1 = torch.empty(0)
            total_AE2 = torch.empty(0)

            for _, x_val in enumerate(loader_valid):
                inp_val = x_val['dat'].to('cuda')
                Y_val = x_val['age'].to('cuda')
                valpredict = (model(inp_val)*torch.linspace(0.5, 18.5, 19).to('cuda')).sum(1)
                valAE1 = (Y_val - valpredict).abs()
                valAE2 = (Y_val - (0.5 + (model(inp_val).max(1)[1]))).abs()
                
                total_AE1 = torch.cat((total_AE1, valAE1.detach().cpu()))
                total_AE2 = torch.cat((total_AE2, valAE2.detach().cpu()))

        valMAE1 = (total_AE1[:total_AE1.shape[0]//2] + total_AE1[total_AE1.shape[0]//2:])/2
        valMAE2 = (total_AE2[:total_AE2.shape[0]//2] + total_AE2[total_AE2.shape[0]//2:])/2

        valMAE1 = valMAE1.mean()
        valMAE2 = valMAE2.mean()

        print('[Epoch: {:>4}] cost = {:>.3}, MAE1 = {:>.3}, MAE2 = {:>.3}'.format(epoch + 1, cost, valMAE1, valMAE2))

        if MAE1tmp > valMAE1:
            MAE1tmp = valMAE1
            if MAE2tmp > valMAE2:
                MAE2tmp = valMAE2
                torch.save(model.state_dict(), 'ENc' + str(ensemble).zfill(2) + '_model_MAE1_'+ str(valMAE1.item()) +'_MAE2_' + str(valMAE2.item()) + '_min.pth')
            else:
                torch.save(model.state_dict(), 'ENc' + str(ensemble).zfill(2) + '_model_MAE1_'+ str(valMAE1.item()) +'_MAE2_' + str(valMAE2.item()) + '.pth')

    model.eval()

files = os.listdir()
result_folder = "./ensemble_child"

for ensemble in range(20):
    min_mae1 = float('inf')
    min_mae1_file = None

    for file in files:
        if file.startswith("ENc" + str(ensemble).zfill(2)) and file.endswith(".pth"):
            mae1_value = float(file.split("_")[3])
            if mae1_value < min_mae1:
                min_mae1 = mae1_value
                min_mae1_file = file
                print(min_mae1)

    if min_mae1_file:
        os.rename(min_mae1_file, os.path.join(result_folder, min_mae1_file))

    for file in files:
        if file.startswith("ENc" + str(ensemble).zfill(2)) and file.endswith(".pth"):
            if file != result_folder and os.path.isfile(file):
                os.remove(file)

#####################################################################

class SFCN(torch.nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.Convblock = nn.Sequential( nn.Conv1d(12, 32, 19, 1, 9),
                                        nn.BatchNorm1d(32),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(32, 64, 19, 1, 9),
                                        nn.BatchNorm1d(64),                        
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                    
                                        nn.Conv1d(64, 128, 19, 1, 9),
                                        nn.BatchNorm1d(128),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(128, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(256, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                        )
        
        self.Convblock2 = nn.Sequential(nn.Conv1d(256, 128, 1, 1, 0),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        )
        
        self.Ave = nn.AdaptiveAvgPool1d(1)

        self.Rest = nn.Sequential(  nn.Dropout(),
                                    nn.Conv1d(128, 103, 1, 1, 0),
                                    )
                                    
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Convblock(x)
        out = self.Convblock2(out)
        out = self.Ave(out)
        out = self.Rest(out)
        out = out.squeeze()
        out = self.Softmax(out)
        return out

class ValidDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        dat = np.load(os.path.join(self.dat_dir, i_name))
        return dat
    
class RealValidDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_names = sorted(os.listdir(self.root_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        i_path = os.path.join(self.root_dir, i_name) 
        dat = np.load(i_path)
        return dat

valid_dataset = ValidDataset(root_dir='./data_train/valid/adult/')
real_valid_dataset = RealValidDataset(root_dir='./data_valid/adult/')

loader_valid = DataLoader(valid_dataset, batch_size=64, shuffle=False)
loader_real_valid = DataLoader(real_valid_dataset, batch_size=64, shuffle=False)

age_val = np.load(dataConcat + 'Adult_train_f.npz')['Y_data'][-valnum:]

totensem = len(os.listdir("./ensemble_adult/"))

total_corrected = 0

for ensemble in range(totensem):
    model = nn.DataParallel(SFCN().to('cuda'))
    model.load_state_dict(torch.load("./ensemble_adult/" + os.listdir("./ensemble_adult/")[ensemble]))
    model.eval()

    total_val = torch.empty(0)
    for x_val in loader_valid:
        inp_val = x_val.to('cuda')
        valpredict = (model(inp_val)*torch.linspace(19.5, 121.5, 103).to('cuda')).sum(1)
        total_val = torch.cat((total_val, valpredict.detach().cpu()))

    valmean = (total_val[:total_val.shape[0]//2] + total_val[total_val.shape[0]//2:])/2

    line_fitter = LinearRegression()
    line_fitter.fit(valmean.reshape(-1, 1), age_val.reshape(-1, 1))

    print('Original: {:>.4}, Corrected = {:>.4}'.format((valmean - age_val).abs().mean(), (((valmean.reshape(-1, 1)*line_fitter.coef_) + line_fitter.intercept_) - age_val.reshape(-1, 1)).abs().mean()))
    
    real_total_val = torch.empty(0)
    for x_real_val in loader_real_valid:
        inp_real_val = x_real_val.to('cuda')
        realvalpredict = (model(inp_real_val)*torch.linspace(19.5, 121.5, 103).to('cuda')).sum(1)
        real_total_val = torch.cat((real_total_val, realvalpredict.detach().cpu()))

    realvalmean = (real_total_val[:real_total_val.shape[0]//2] + real_total_val[real_total_val.shape[0]//2:])/2
    total_corrected += (realvalmean.reshape(-1, 1)*line_fitter.coef_) + line_fitter.intercept_

total_corrected /= totensem

total_predicted = np.zeros(total_adult_val)
OODs = 0
for i in range(total_adult_val):
    if i in adult_valid_nans:
        total_predicted[i] = (19 + 122)/2
        OODs += 1
    else:
        total_predicted[i] = total_corrected[i-OODs]

np.savetxt("predicted_adult.csv", total_predicted, delimiter=",")

##########################################
class SFCN(torch.nn.Module):
    def __init__(self):
        super(SFCN, self).__init__()
        self.Convblock = nn.Sequential( nn.Conv1d(12, 32, 19, 1, 9),
                                        nn.BatchNorm1d(32),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(32, 64, 19, 1, 9),
                                        nn.BatchNorm1d(64),                        
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                    
                                        nn.Conv1d(64, 128, 19, 1, 9),
                                        nn.BatchNorm1d(128),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(128, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),

                                        nn.Conv1d(256, 256, 19, 1, 9),
                                        nn.BatchNorm1d(256),
                                        nn.MaxPool1d(2, 2),
                                        nn.ReLU(),
                                        )
        
        self.Convblock2 = nn.Sequential(nn.Conv1d(256, 64, 1, 1, 0),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        )
        
        self.Ave = nn.AdaptiveAvgPool1d(1)

        self.Rest = nn.Sequential(  nn.Dropout(),
                                    nn.Conv1d(64, 19, 1, 1, 0),
                                    )
                                    
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.Convblock(x)
        out = self.Convblock2(out)
        out = self.Ave(out)
        out = self.Rest(out)
        out = out.squeeze()
        out = self.Softmax(out)
        return out
    
class ValidDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dat_dir = os.path.join(self.root_dir, 'dat')
        self.input_names = sorted(os.listdir(self.dat_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        dat = np.load(os.path.join(self.dat_dir, i_name))
        return dat
    
class RealValidDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_names = sorted(os.listdir(self.root_dir))

    def __len__(self): 
        return len(self.input_names)

    def __getitem__(self, idx):
        i_name = self.input_names[idx]
        i_path = os.path.join(self.root_dir, i_name) 
        dat = np.load(i_path)
        return dat

valid_dataset = ValidDataset(root_dir='./data_train/valid/child/')
real_valid_dataset = RealValidDataset(root_dir='./data_valid/child/')

loader_valid = DataLoader(valid_dataset, batch_size=64, shuffle=False)
loader_real_valid = DataLoader(real_valid_dataset, batch_size=64, shuffle=False)

age_val = np.load(dataConcat + 'Child_train_f.npz')['Y_data'][-valnum:]

totensem = len(os.listdir("./ensemble_child/"))

total_corrected = 0

for ensemble in range(totensem):
    model = nn.DataParallel(SFCN().to('cuda'))
    model.load_state_dict(torch.load("./ensemble_child/" + os.listdir("./ensemble_child/")[ensemble]))
    model.eval()

    total_val = torch.empty(0)
    for x_val in loader_valid:
        inp_val = x_val.to('cuda')
        valpredict = (model(inp_val)*torch.linspace(0.5, 18.5, 19).to('cuda')).sum(1)
        total_val = torch.cat((total_val, valpredict.detach().cpu()))

    valmean = (total_val[:total_val.shape[0]//2] + total_val[total_val.shape[0]//2:])/2

    line_fitter = LinearRegression()
    line_fitter.fit(valmean.reshape(-1, 1), age_val.reshape(-1, 1))

    print('Original: {:>.4}, Corrected = {:>.4}'.format((valmean - age_val).abs().mean(), (((valmean.reshape(-1, 1)*line_fitter.coef_) + line_fitter.intercept_) - age_val.reshape(-1, 1)).abs().mean()))
    
    real_total_val = torch.empty(0)
    for x_real_val in loader_real_valid:
        inp_real_val = x_real_val.to('cuda')
        realvalpredict = (model(inp_real_val)*torch.linspace(0.5, 18.5, 19).to('cuda')).sum(1)
        real_total_val = torch.cat((real_total_val, realvalpredict.detach().cpu()))

    realvalmean = (real_total_val[:real_total_val.shape[0]//2] + real_total_val[real_total_val.shape[0]//2:])/2
    total_corrected += (realvalmean.reshape(-1, 1)*line_fitter.coef_) + line_fitter.intercept_

total_corrected /= totensem

total_predicted = np.zeros(total_child_val)
OODs = 0
for i in range(total_child_val):
    if i in child_valid_nans:
        total_predicted[i] = (0 + 19)/2
        OODs += 1
    else:
        total_predicted[i] = total_corrected[i-OODs]

np.savetxt("predicted_child.csv", total_predicted, delimiter=",")
