import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable



class Seg_CNN(torch.nn.Module):
    def __init__(self,img_size):
        super(Seg_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(img_size*img_size,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, data):
        img = data['image']
        img_size = img.size()[3]
        # print("img size {}".format(img.size()))
        #shape of x is (b_s, 32,32,1)
        img = self.conv1(img) #shape of x is (b_s, 28,28,32)
        # print("img size {}".format(img.size()))
        img = F.relu(img)
        img = self.pool1(img) #shape of x now becomes (b_s X 14 x 14 x 32)

        img = self.conv2(img) # shape(b_s, 10x10x64)
        img = F.relu(img)
        img = self.pool2(img)
        # print("img size {}".format(img.size()))

        img = self.conv3(img) # shape(b_s, 10x10x64)
        img = F.relu(img)        
        img = self.pool3(img)
        # print("img size {}".format(img.size()))

        # img = F.relu(img) #size is (b_s x 10 x 10 x 64)
        img = img.view(-1, img_size*img_size) # shape of x is now(b_s*2, 3200)
        # print("img size {}".format(img.size()))

        #fc1 to be of shape (6400,1024)
        img = self.fc1(img)
        img = F.relu(img)
        img = self.fc2(img)
        img = F.relu(img)
        img = self.fc3(img)
        return img 


class VGG16(torch.nn.Module):
    
    def __init__(self):
        super(VGG16, self).__init__()
        
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(1, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112
        
        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
        
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
        
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        
        # view        
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.fc4 = nn.Linear(1000, 1)


    def forward(self, x):
        
        img = x['image']
        # x.size(0)即为batch_size
        in_size = img.size()[0]
        
        out = self.conv1_1(img) # 222
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7
        
        # 展平
        out = out.view(in_size, -1)
        # print("img size {}".format(out.size()))

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)

        out = self.fc4(out)

        # out = F.log_softmax(out, dim=1)

        return out


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res