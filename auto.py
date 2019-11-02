# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importing the dataset
movies = pd.read_csv('ml-latest/ml-latest/movies.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-latest/ml-latest/tags.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-latest/ml-latest/ratings.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set_AUTO = pd.read_csv('ml-100k/ml-100k/u2.base', delimiter = '\t')
training_set_AUTO  = np.array(training_set_AUTO , dtype = 'int')
test_set_AUTO  = pd.read_csv('ml-100k/ml-100k/u2.test', delimiter = '\t')
test_set_AUTO  = np.array(test_set_AUTO , dtype = 'int')

# Getting the number of users and movies
total_number_of_users = int(max(max(training_set_AUTO[:,0]), max(test_set_AUTO[:,0])))
total_number_of_movie = int(max(max(training_set_AUTO[:,1]), max(test_set_AUTO[:,1])))

#we create the data into an array with lines in users and colomns in movies
def convert(data_100K):
    new_data_100K = []
    for AUTO_users_100K in range(1, total_number_of_users+ 1):
        AUTO_movies_100K = data_100K[:,1][data_100K[:,0] == AUTO_users_100K]
        AUTO_ratings_100K = data_100K[:,2][data_100K[:,0] == AUTO_users_100K]
        ratings = np.zeros(total_number_of_movie)
        ratings[AUTO_movies_100K - 1] = AUTO_ratings_100K
        new_data_100K.append(list(ratings))
    return new_data_100K
training_set_AUTO = convert(training_set_AUTO)
test_set_AUTO = convert(test_set_AUTO)

# Converting the data into Torch tensors
training_set_AUTO= torch.FloatTensor(training_set_AUTO)
test_set_AUTO  = torch.FloatTensor(test_set_AUTO)

#CREATE THE AUTOENCODER NETWORK
#WE USE INHERITANCE BY CREATING A CHILD CLASS FROM THE PARENT CLASS (module)
#WE ARE USING A STACKED AUTOENCODER
class AutoEncoder(nn.Module):
    def __init__(self, ):
        #WE GET THE INHERITED METHODS FROM THE MODULE CLASS
        #WE USE SUPER TO USE THE METHODS AND CLASSES FROM THE nn Module
        super( AutoEncoder, self).__init__()
        #FULL CONNECTION BETWEEN THE FIRST INPUT VECTOR FEATURES AND THE FIRST ENCODED VECTOR
        self.fulconnect1 = nn.Linear(total_number_of_movie, 30)
        #FULL CONNECTION BETWEEN THE SECOND INPUT VECTOR FEATURES AND THE SECOND ENCODED VECTOR
        self.fulconnect2 = nn.Linear(30,20)
        #FULL CONNECTION BETWEEN THE THRID INPUT VECTOR FEATURES AND THE THIRD ENCODED VECTOR
        self.fulconnect3 = nn.Linear(20,10)
        #WE BEGIN THE DECODING
        self.fulconnect4 = nn.Linear(10,20)
        self.fulconnect5 = nn.Linear(20,30)
        self.fulconnect6 = nn.Linear(30,total_number_of_movie)
        #INTRODUCE SIGMOID ACTIVATION FUNCTION
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation( self.fulconnect1(x))
        x = self.activation( self.fulconnect2(x))
        x = self.activation( self.fulconnect3(x))
        x = self.activation( self.fulconnect4(x))
        x = self.activation( self.fulconnect5(x))
        x = self.fulconnect6(x)
        return x

stacked = AutoEncoder()
criterion = nn.MSELoss() #CRITERION FOR THE LOSS FUNCTION(MEAN SQUARED ERROR)
#USE OPTIMIZER THAT APPLIES STOCASTIC GRADIENT DESCENT TO UPDATE THE WEIGHTS
optimizer = optim.RMSprop(stacked.parameters(), lr = 0.01, weight_decay = 0.5)

#Evakuation of AutoEncoder model 
# Training the AutoEncoder
number_of_epoch = 200
for epoch in range(1, number_of_epoch + 1):
    train_loss = 0
    s = 0. #number of users that rated at least one movie, for optomization
    for AUTO_users_100K in range(total_number_of_users):
        input = Variable(training_set_AUTO[AUTO_users_100K]).unsqueeze(0)#WE CREATE EXTRA DIMENSION(BATCH)
        target_AUTO = input.clone()
        if torch.sum(target_AUTO.data > 0) > 0:
            output = stacked(input)
            target_AUTO.require_grad = False
            output[target_AUTO == 0] = 0
            loss = criterion(output, target_AUTO)
            mean_corrector = total_number_of_movie/float(torch.sum(target_AUTO.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the AutoEncoder
test_loss = 0
s = 0.
for AUTO_users_100K in range(total_number_of_users):
    input = Variable(training_set_AUTO[AUTO_users_100K]).unsqueeze(0)
    target = Variable(test_set_AUTO[AUTO_users_100K])
    if torch.sum(target_AUTO.data > 0) > 0:
        output = stacked(input)
        target_AUTO.require_grad = False
        output[target_AUTO == 0] = 0
        loss = criterion(output, target_AUTO)
        mean_corrector = total_number_of_movie/float(torch.sum(target_AUTO.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
