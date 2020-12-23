import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon
import torch
from torch_geometric.data import Data, Dataset,DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
import csv
import argparse
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
from model_test import TbNet

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz,. "
vob = {x:ind for ind, x in enumerate(alphabet)}

def encode_text(ins, vob, max_len = 10, default = " "):
    out = []
    sl = len(ins)
    minl = min(sl, max_len)
    for i in range(minl):
        char = ins[i]
        if char in vob:
            out.append(vob[char])
        else:
            out.append(vob[default])
    if len(out)<=max_len:
        out = out +[vob[default]]*(max_len-len(out))
    return out

class ValidateDataset(Dataset):
    def __init__(self, root_path, transform=None, pre_transform=None):
        super(ValidateDataset, self).__init__(root_path, transform, pre_transform)
        self.root_path = root_path
        self.jsonfile = os.path.join(self.root_path, "validateImgList.json")
        self.img_size = 256
        self.kernel = np.ones((3,3),np.uint8)  # 把图像的线变粗一点
        if os.path.exists(self.jsonfile):  # imglist.json去掉了一些有疑问的文件
            with open(self.jsonfile, "r") as read_file:
                self.imglist = json.load(read_file)
        else:  
            self.imglist = list(filter(lambda fn:fn.lower().endswith('.jpg') or fn.lower().endswith('.png') ,
                                       os.listdir(os.path.join(self.root_path,"img"))))
            self.imglist = self.check_all()
            with open(self.jsonfile, "w") as write_file:
                json.dump(self.imglist, write_file)
        self.graph_transform = GT.KNNGraph(k=6)
     
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
    
    def read_structure(self):
        return 
        
    def reset(self):
        pass
    
    def check_all(self):
        validlist=[]
        for idx in range(len(self.imglist)):
            print("*** file:",self.imglist[idx])
            chunks ,img = self.readlabel(idx)
            validlist.append(self.imglist[idx])
        print("valid:", len(validlist))
        return validlist
    
    def readlabel(self,idx):
        imgfn = self.imglist[idx]
        chunkfn = os.path.join(self.root_path,"chunk",os.path.splitext(os.path.basename(imgfn))[0]+".chunk")
        imgfn = os.path.join(self.root_path,"img",os.path.splitext(os.path.basename(imgfn))[0]+".png")
        if not os.path.exists(chunkfn) or not os.path.exists(imgfn):
            print("can't find files.")
            return {}, {}
        with open(chunkfn, 'r', encoding="utf8") as f:
            chunks = json.load(f)['chunks']
        if len(chunks) == 0:
            print(chunkfn)
        img = cv2.imread(imgfn)
        if not img is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = 255 - img
            img = cv2.dilate(img,self.kernel,iterations = 1)
            img = cv2.resize(img, (self.img_size,self.img_size), interpolation = cv2.INTER_AREA) 
        
        return chunks ,img 
        
    
    def __len__(self):
        return len(self.imglist)
    
    
    def box_center(self, chkp):
        # x1, x2, y1, y2  in chkp
        return [(chkp[0]+chkp[1])/2, (chkp[2]+chkp[3])/2]
        
    def get_html(self, idx):
        structs, chunks, img = self.readlabel(idx)
        self.check_chunks(structs,chunks)
        html = self.format_html(structs, chunks)
        return html
    
    
    def cal_chk_limits(self, chunks):
        x_min = min(chunks, key=lambda p: p["pos"][0])["pos"][0]
        x_max = max(chunks, key=lambda p: p["pos"][1])["pos"][1]
        y_min = min(chunks, key=lambda p: p["pos"][2])["pos"][2]
        y_max = max(chunks, key=lambda p: p["pos"][3])["pos"][3]
        hlist = [p["pos"][3]-p["pos"][2] for p in chunks] 
        avhei = sum(hlist)/len(hlist)
        # 加入一点边界, 大概对应整个图像。
        width = x_max-x_min + 2*avhei
        height = y_max-y_min + 0.5*2*avhei
        return [x_min,x_max,y_min,y_max,width,height,avhei] # 
    
    # 相对的位置。
    def pos_feature(self,chk,cl): 
        x1=(chk["pos"][0]-cl[0]+cl[6])/cl[4] 
        x2=(chk["pos"][1]-cl[0]+cl[6])/cl[4] 
        x3=(chk["pos"][2]-cl[2]+0.5*cl[6])/cl[5] 
        x4=(chk["pos"][3]-cl[2]+0.5*cl[6])/cl[5]
        x5 = (x1+x2)*0.5  # 中心点
        x6 = (x3+x4)*0.5
        x7 = x2-x1    # 文本宽度
        x8 = x4-x3    # 文本高度
        return [x1,x2,x3,x4,x5,x6,x7,x8]
    
    def augmentation_chk(self, chunks):
        for chk in chunks:
            chk["pos"][0] += random.normalvariate(0,1)
            chk["pos"][1] += random.normalvariate(0,1)
            chk["pos"][2] += random.normalvariate(0,1)
            chk["pos"][3] += random.normalvariate(0,1)
        
    def get(self, idx):
        chunks, img = self.readlabel(idx)
        
        cl = self.cal_chk_limits(chunks)
        x,pos,tbpos,xtext,imgpos=[],[],[],[],[]
        plaintext = []
        for chk in chunks: 
            xt = self.pos_feature(chk,cl)
            x.append(xt)            
            pos.append(xt[4:6]) # center point
            xtext.append(encode_text(chk["text"],vob))
            plaintext.append(chk["text"].encode('utf-8'))
            imgpos.append([(1.0-xt[5])*2-1.0, xt[4]*2-1.0]) # 图像中的y是倒过来的。这是归一化[-1,1]之间。图像的y在前，和H对应。
                
            
        x = torch.FloatTensor(x)  
        pos = torch.FloatTensor(pos)  
        data = Data(x=x,pos=pos)
        data = self.graph_transform(data) # 构造图的连接
        img = torch.FloatTensor(img/255.0).unsqueeze(0).unsqueeze(0)
        data.img = img
        data.imgpos = torch.FloatTensor(imgpos)
        data.nodenum = torch.LongTensor([len(chunks)])
        data.xtext = torch.LongTensor(xtext)
        return data
    
    def cal_label(self,data,tbpos): # 根据构造的图，计算边的标注。
        edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y = []
        for i in range(edges.size()[1]):
            # 计算同列还是同行关系
            # y.append(self.if_same_row(edges[0,i], edges[1,i],tbpos))
            y.append(self.if_same_col(edges[0,i], edges[1,i],tbpos))
        return y
            
    def if_same_row(self,si,ti,tbpos):
        ss,se = tbpos[si][0], tbpos[si][1]
        ts,te = tbpos[ti][0], tbpos[ti][1]
        if (ss>=ts and se<=te):
            return 1
        if (ts>=ss and te<=se):
            return 1
        return 0
    
    def if_same_col(self,si,ti,tbpos):
        ss,se = tbpos[si][2], tbpos[si][3]
        ts,te = tbpos[ti][2], tbpos[ti][3]
        if (ss>=ts and se<=te):
            return 1
        if (ts>=ss and te<=se):
            return 1
        return 0
        
    
    def if_same_cell(self):
        pass


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = DataLoader(dataset, batch_size=32)
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        preds = net(data)
        _, preds = preds.max(1)
        preds = preds.detach().cpu().numpy()

        edges_graph = data.edge_index.numpy()
        # print(preds)
        # print(edges_graph)

        # TODO：在这里整理预测出来的结果，变成直接可视的表格结构
        # sameColMatrix = ouputSameCol(preds, edges_graph)
        # samecols = getSameColSet(sameColMatrix)

        # print("the predict is: {}".format(samecols))

# def getSameColSet(pairs):
#     res = []
#     for i in range(len(pairs[1])):
#         node1 = pairs[0][i]
#         node2 = pairs[1][i]
#         if node1==node2:
#             continue

#         index1 = -1
#         index2 = -1
#         for j in range(len(res)):
#             if node1 in res[j]:
#                 index1 = j
#             if node2 in res[j]:
#                 index2 = j

#         if index1 == index2:
#             if index2 == -1:
#                 newset = {node1,node2}
#                 res.append(newset)
#             else:
#                 res[index1].add(node1)
#                 res[index1].add(node2)
#         else:
#             if index1==-1 or index2==-1:
#                 index_put = index2 if index1==-1 else index1
#                 res[index_put].add(node1)
#                 res[index_put].add(node2)
#             else:
#                 index_keep = min(index1, index2)
#                 index_remove = max(index1, index2)
#                 res[index_keep].union(res[index_remove])
#                 res.pop(index_remove)
#     print(res)


# def ouputSameCol(preds, edges):
#     indexs = np.where(preds==0)
#     output = np.delete(edges, indexs, axis=1)
#     print("ouputSameCol is {}".format(output))
#     n = len(output[0])
#     for i in range(n):
#         print("{} - {}".format(output[0][i],output[1][i]))
#     return output


def printData(root_path):
    ds = ValidateDataset(root_path)

    test_loader = DataLoader(ds, batch_size=5 )

    for data in test_loader:
        print(data,data.num_graphs)
        print("data.x:{}".format(data.x))
        print(data.imgpos[0:10,:], data.nodenum)
        torch.save(data, "temp.pt")


def predicateData(data_path, model_path):
    test_dataset = ValidateDataset(data_path)

    device = torch.device("cpu" )
    model = TbNet(input_num,vocab_size,num_text_features,nclass)
    model.cpu()
    criterion = torch.nn.NLLLoss() 

    model.load_state_dict(torch.load(model_path))

    val(model, test_dataset, criterion)

nclass = 2
input_num = 8
vocab_size = 39
num_text_features = 64

# 更改路径
data_path = r'C:\Users\zhak\GFTE\testTable\data'
rcnnPath = r"C:\Users\zhak\GFTE\expr\net_20_360.pth"

if __name__ == "__main__":  
    # printData(data_path)

    predicateData(data_path, rcnnPath)
