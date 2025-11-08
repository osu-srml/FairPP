"""Loading data from file"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from retention import best_response
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from folktables import ACSDataSource, ACSIncome

# Define a custom dataset that separates the digits into two groups
# class CustomMNISTDataset(Dataset):
#     def __init__(self, root, train=True, transform=None):
#         self.mnist_data = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)
#         self.transform = transform

#     def __len__(self):
#         return len(self.mnist_data)

#     def __getitem__(self, idx):
#         image, label = self.mnist_data[idx]
#         # Divide the digits into two groups (0-4 and 5-9)
#         if label < 5:
#             label_group = 0  # Group for digits 0-4
#         else:
#             label_group = 1  # Group for digits 5-9

#         if self.transform:
#             image = self.transform(image)

#         return image, label_group


def MNIST_data(p_0=0.3, p_1=0.7, num_samples=1000,seed=0,batch_size=64):
    """
    load MNIST data and divide into 2 groups
    The first group is digits 0-4, the second group is digits 5-9
    """
    # define transformation
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    custom_dataset = datasets.MNIST('../data', train=True, download=True,transform=transform)
    n0 = int(num_samples*p_0)
    n1 = num_samples - n0
    indices_0= np.where(np.array(custom_dataset.targets) <= 4)[0]
    indices_1 = np.where(np.array(custom_dataset.targets) > 4)[0]
    indices_0 = np.random.choice(indices_0, size=n0, replace=False)
    indices_1 = np.random.choice(indices_1, size=n1, replace=False)
    # Combine the indices for both groups
    selected_indices = np.concatenate((indices_0,indices_1))
    # Create a sampler for the train loader
    train_sampler = SubsetRandomSampler(selected_indices)
    # Create a data loader with the custom sampler
    train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=train_sampler)
    return train_loader

    

def Credit_data(seed=0):
    """Load data from csv file.

    Parameters
    ----------
        file_loc: string
            path to the '.csv' training data file
    Returns
    -------
        X_full: np.array
            balances data matrix     
        Y_full: np.array
            corresponding labels (0/1) 
        data: DataFrame
            raw data     
    """
    np.random.seed(seed)
    data_all = pd.read_csv('cs-training.csv', index_col=0)
    data_all.dropna(inplace=True)
    data = data_all.sample(frac=0.1, replace=False, random_state=seed)
    data['Z'] = np.where(data['age'] > 50, 0, 1)

    # full data set
    X_all = data.drop('SeriousDlqin2yrs', axis=1)
    # X_all = data[['RevolvingUtilizationOfUnsecuredLines','DebtRatio', 'MonthlyIncome']]

    # zero mean, unit variance
    X_all = preprocessing.scale(X_all)
    X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)


    # outcomes
    Y_all = np.array(data['SeriousDlqin2yrs'])

    # sensitive attributes
    Z_all = np.array(data['Z'])

    # balance classes
    default_indices = np.where(Y_all == 1)[0][:1000]
    other_indices = np.where(Y_all == 0)[0][:1000]
    indices = np.concatenate((default_indices, other_indices))

    X_balanced = X_all[indices]
    Y_balanced = Y_all[indices]
    Z_balanced = Z_all[indices]

    # shuffle arrays
    p = np.random.permutation(len(indices))
    X_full = X_balanced[p]
    Y_full = Y_balanced[p]
    Z_full = Z_balanced[p]
    return X_full, Y_full, Z_full, data

def sample_credit_data(p_0, p_1, num_samples,seed = 0):
    np.random.seed(seed)
    X_full, Y_full, Z_full, _ = Credit_data()
    n_0 = int(num_samples * p_0)
    z0_idx = np.where(Z_full == 0)[0]
    z1_idx = np.where(Z_full == 1)[0]
    z0_choice = np.random.choice(z0_idx, size=n_0, replace=True)
    z1_choice = np.random.choice(z1_idx, size=num_samples-n_0, replace=True)
    indices = np.concatenate((z0_choice,z1_choice))
    X, Y, Z = X_full[indices], Y_full[indices], Z_full[indices]
    p = np.random.permutation(len(indices))
    return X[p], Y[p], Z[p]

def Income_data(seed=0):
    np.random.seed(seed)
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = pd.read_csv('ca_data.csv')

    ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)

    ca_data = pd.concat([ca_features,ca_labels],axis=1)

    ca_data['PINCP'] = ca_data['PINCP'].map({True: 1, False: 0}).astype(int)
    df = ca_data
    indices = np.random.choice(len(df), 5000, replace=False)
    df = df.iloc[indices]
    Features = ['AGEP','PINCP','SCHL','WKHP']

    # df = pd.get_dummies(df, columns=CategoricalFeatures, drop_first=True)
    df = df[Features]
    df = df.rename(columns = {'PINCP':'y'})
    Y_all = np.array(df['y'])
    X_all = df.drop(columns=['y'])
    X_all = preprocessing.scale(X_all)
    X_all = np.append(X_all, np.ones((X_all.shape[0], 1)), axis=1)
    Z_all = np.array(np.where(df['AGEP'] < 35, 0.0, 1.0))
    return X_all, Y_all, Z_all, df

def sample_income_data(p_0, p_1, num_samples,seed = 0):
    np.random.seed(seed)
    X_full, Y_full, Z_full, _ = Income_data()
    n_0 = int(num_samples * p_0)
    z0_idx = np.where(Z_full == 0)[0]
    z1_idx = np.where(Z_full == 1)[0]
    z0_choice = np.random.choice(z0_idx, size=n_0, replace=True)
    z1_choice = np.random.choice(z1_idx, size=num_samples-n_0, replace=True)
    indices = np.concatenate((z0_choice,z1_choice))
    X, Y, Z = X_full[indices], Y_full[indices], Z_full[indices]
    p = np.random.permutation(len(indices))
    return X[p], Y[p], Z[p]
    

def Gaussian_data(x_dis = {'mean': (0.5,0.5), 'cov': np.array([[0.25,0.0], [0.0,0.25]])}, y_dis = {0:(1,-0.5), 1: (0.5,0.5)}, p_0 = 0.3, p_1 = 0.7,num_samples=10000, regression=False, seed=0):
    """
    Make a Gaussian synthetic dataset consisting of 2 groups
    Each group has 2 features, with same means and standard deviations
    However, the relationship y|x is different
    also different initial group proportions

    """
    np.random.seed(seed)
    xs, ys, zs = [], [], []
    for i in range(num_samples):
        z = np.random.binomial(n = 1, p = p_1, size = 1)[0]
        x = np.random.multivariate_normal(mean = x_dis['mean'], cov = x_dis['cov'], size = 1)[0]
        x = np.append(x,1)
        y = y_dis[z][0]*x[0] + y_dis[z][1]*x[1] + np.random.normal(0, 0.05) # add a noise to the label
        if not regression:
            if y > 0.5:
                y = 1
            else:
                y = 0
        xs.append(x)
        ys.append(y)
        zs.append(z)
    
    data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], np.array(xs).T[2], ys, zs), columns = ['x1', 'x2', 'intercept','y', 'z'])
    X_all = data.drop(['y','z'], axis=1)
    # X_all['inter'] = np.ones((X_all.shape[0], 1))
    Y_all = np.array(data['y'])
    Z_all = np.array(data['z'])
    return X_all, Y_all, Z_all, data
    


def Gaussian_mean_data(mu_1 = 0.3, mu_2 = 0.7, sd = 0.05, p_0 = 0.3, p_1 = 0.7,num_samples=10000,seed=0):
    """
    1-d for gaussian mean estimation
    """

    np.random.seed(seed)
    zs = np.random.binomial(n = 1, p = p_1, size = num_samples)
    xs = np.ones(num_samples).reshape(-1,1)
    ys = np.where(zs==0,mu_1,mu_2)
    ys += np.random.normal(loc=0,scale=sd,size=num_samples)
    return xs,ys,zs


def Gaussian_mean_data_multi(mu_list = [0.3,0.5,0.7], sd = 0.05, p_list = [0.1,0.3,0.6],num_samples=10000,seed=0):
    """
    1-d for gaussian mean estimation
    """

    np.random.seed(seed)
    multinomial_vars = np.random.multinomial(num_samples, p_list)
    zs = np.repeat(list(range(len(p_list))),multinomial_vars)
    zs = np.random.permutation(zs)
    xs = np.ones(num_samples).reshape(-1,1)
    ys = np.zeros(num_samples)
    for z in range(len(p_list)):
        ys[zs==z] = mu_list[z]
    ys += np.random.normal(loc=0,scale=sd,size=num_samples)
    return xs,ys,zs



