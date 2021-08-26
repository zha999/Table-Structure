import torch.optim as optim
import torch.nn as nn
from data import Seg_Dataset
from model import Seg_CNN, averager, VGG16
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.metrics.functional import r2score
from sklearn.metrics import r2_score
import torch
import os
import csv 
import numpy as np

# options
epoch_num = 31 # 101
prompt_interval = 5
validate_interval = 1
save_interval = 1

learning_rate = 0.001
beta_adam = 0.5

batch_length = 32
result_file = 'exp'
os.system('mkdir {0}'.format(result_file))

img_size = 64 # 224

# data set
# root_path = r'E:\cnn_segment\dataset_att\generated' #
root_path = r'E:\cnn_segment\dataset_add'
train_path = os.path.join(root_path, 'Train')
#r'E:\cnn_segment\dataset_add\Train'
train_dataset = Seg_Dataset(train_path, img_size)
test_path = os.path.join(root_path, 'Test') 
# r'E:\cnn_segment\dataset_add\Test'
test_dataset = Seg_Dataset(test_path, img_size)

train_loader = DataLoader(train_dataset, batch_size=batch_length, shuffle=True)

# model
# model = VGG16() # Seg_CNN()
model = Seg_CNN(img_size)
# model.cuda()

mse_loss = nn.MSELoss()
# r2_loss = r2_score
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_adam, 0.999))

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


res_headers = ["epoch", "loss", "r2"]
val_res_1 = []
val_res_2 = []

def val(net, dataset, val_type, epoch = 0, max_iter=100):
    print('start val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = DataLoader(dataset, batch_size=batch_length)
    val_iter = iter(data_loader)

    data_len = len(data_loader)
    print("data_loader size {}".format(data_len))

    loss_avg = averager()
    mse_avg = averager()
    mse_avg.reset()
    max_iter = min(max_iter, data_len)

    # all_pre = torch.Tensor()    
    # all_data = torch.Tensor() 

    all_pre = np.array([])   
    all_data = np.array([])

    for i in range(data_len):
        data = val_iter.next()

        preds = net(data)
        preds = preds.squeeze(1)
        # print("preds size {}".format(preds.size()))
        data = data['att'].cpu()
        # print("data size {}".format(data.size()))

        mse_cost = mse_loss(preds, data) 
        mse_avg.add(mse_cost)

        # if i==0:
        #     all_pre = preds
        #     all_data = data
        # else:
        #     all_pre = torch.cat((all_pre, preds))
        #     all_data = torch.cat((all_data, data))
        all_pre = np.concatenate((all_pre, preds),axis=0)
        all_data = np.concatenate((all_data, data),axis=0)
        # print("all_pre size {}".format(all_pre.size()))
        # print("all_data size {}".format(all_data.size()))

        # r2_cost = r2_loss(preds, data['att'].cpu()) 
        # loss_avg.add(r2_cost)

        # print("1-mse_avg: {}".format(mse_avg.val()))
        # print("1-loss_avg: {}".format(loss_avg.val()))
    
    print("all_pre size {}".format(all_pre.shape))
    print("all_data size {}".format(all_data.shape))
    print("all_pre is {}".format(all_pre))
    print("all_data is {}".format(all_data))
    # r2_value = r2_loss(all_pre, all_data)
    r2_value = r2_score(all_data, all_pre)

    
    print('mse_avg: %f, r2_value: %f' % (mse_avg.val(), r2_value))

    # collect results:
    if val_type == 1:
        val_res_1.append([epoch, mse_avg.val(), r2_value])
    if val_type == 2:
        val_res_2.append([epoch, mse_avg.val(), r2_value])

def trainBatch(train_iter, net, optimizer):
    data = train_iter.next()
    preds = net(data).squeeze(1)
    #batch_size = data.y.size()[0]
    cost = mse_loss(preds, data['att'].cpu()) 
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


# validate first
print("On test Eval:")
# val(model, test_dataset,2)

loss_avg = averager()
# training
for epoch in range(epoch_num):
    train_iter = iter(train_loader)
    i = 0
    print('epoch',epoch, ' dataset size:', len(train_loader))
    train_loss = 0

    model.train()

    while i < len(train_loader):
        for p in model.parameters():
            p.requires_grad = True
        # model.train()
  
        # cost = trainBatch(train_iter, model, optimizer)
        # loss_avg.add(cost)

        data = train_iter.next()

        optimizer.zero_grad()
        preds = model(data).squeeze(1)
        cost = mse_loss(preds, data['att']) 
        cost.backward()
        optimizer.step()

        train_loss += cost

        i += 1
        #print(loss_avg)
        print('[%d/%d][%d/%d] Loss: %f' % (epoch, epoch_num, i, len(train_loader), cost))
        # if i % prompt_interval == 0:
            # print('[%d/%d][%d/%d] Loss: %f' %
            #       (epoch, epoch_num, i, len(train_loader), cost))
            # loss_avg.reset()

    if epoch % validate_interval == 0:
        print("On Train:")
        val(model, train_dataset, 1, epoch)
        print("On Test:")
        val(model, test_dataset, 2, epoch)

    # do checkpointing
    if epoch % save_interval == 0 :
        torch.save(model.state_dict(), '{0}/net_{1}_{2}.pth'.format(result_file, epoch, i))


with open("results1.csv", 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(res_headers)
    f_csv.writerows(val_res_1)

with open("results2.csv", 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(res_headers)
    f_csv.writerows(val_res_2)