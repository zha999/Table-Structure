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
import copy

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
            chunks ,img = self.readlabel(idx)   # structs,
            validlist.append(self.imglist[idx])
        print("valid:", len(validlist))
        return validlist

    
    def format_html(self,structs, chunks):
        rowcnt = max(structs, key=lambda p: p["end_row"])["end_row"]+1
        colcnt = max(structs, key=lambda p: p["end_col"])["end_col"]+1
        #print("row , col number:", rowcnt, colcnt)
        mat = [["<td></td>"]*colcnt for i in range(rowcnt)]
        for st in structs: # 填空
            mat[st["start_row"]][st["start_col"]] = "<td>" + st["tex"] + "</td>"
        html = ""
        #print(mat)
        for row in mat:
            html += "<tr>"+"".join(row)+"</tr>"
        return html    
        
    
    def readlabel(self,idx):
        imgfn = self.imglist[idx]
        chunkfn = os.path.join(self.root_path,"chunk",os.path.splitext(os.path.basename(imgfn))[0]+".chunk")
        imgfn = os.path.join(self.root_path,"img",os.path.splitext(os.path.basename(imgfn))[0]+".png")
        if not os.path.exists(chunkfn) or not os.path.exists(imgfn): # not os.path.exists(structfn) or
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
        
        for chk in chunks: # structs
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
        data.nodenum = torch.LongTensor([len(chunks)]) # structs
        data.xtext = torch.LongTensor(xtext)
        return data
    
    def cal_label(self,data,tbpos): # 根据构造的图，计算边的标注。
        edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y = []
        for i in range(edges.size()[1]):
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

class Validate_RowDataset(Dataset):
    def __init__(self, root_path, transform=None, pre_transform=None):
        super(Validate_RowDataset, self).__init__(root_path, transform, pre_transform)
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
            chunks ,img = self.readlabel(idx)   # structs,
            validlist.append(self.imglist[idx])
        print("valid:", len(validlist))
        return validlist
    
    def readlabel(self,idx):
        imgfn = self.imglist[idx]
        chunkfn = os.path.join(self.root_path,"chunk",os.path.splitext(os.path.basename(imgfn))[0]+".chunk")
        imgfn = os.path.join(self.root_path,"img",os.path.splitext(os.path.basename(imgfn))[0]+".png")
        if not os.path.exists(chunkfn) or not os.path.exists(imgfn): # not os.path.exists(structfn) or
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
        
        for chk in chunks: # structs
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
        data.nodenum = torch.LongTensor([len(chunks)]) # structs
        data.xtext = torch.LongTensor(xtext)
        return data
    
    def cal_label(self,data,tbpos): # 根据构造的图，计算边的标注。
        edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y = []
        for i in range(edges.size()[1]):
            y.append(self.if_same_row(edges[0,i], edges[1,i],tbpos))
            # y.append(self.if_same_col(edges[0,i], edges[1,i],tbpos))
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


def val_cols(net_col, dataset, criterion, max_iter=100):
    print('---------->>>Start val col')
    for p in net_col.parameters():
        p.requires_grad = False
    net_col.eval()

    data_loader = DataLoader(dataset, batch_size=1)
    val_iter = iter(data_loader)

    # for i in range(max_iter):
    data = val_iter.next()
    preds_col = net_col(data)
    _, preds_col = preds_col.max(1)
    preds_col = preds_col.detach().cpu().numpy()

    edges_graph = data.edge_index.numpy()
    # print(type(preds_col))
    # print(preds_col)
    # print(type(edges_graph))
    # print(edges_graph)

    sameColMatrix = ouputSameCol(preds_col, edges_graph, data.x)
    sameCols = getClusterSet(sameColMatrix)
    # print("the predict is: {}".format(sameCols))

    appendList = []
    colNodes = set()
    for i in sameCols:
        colNodes = colNodes | i
    for i in range(len(data.x)):
        if i not in colNodes:
            appendList.append({i})
    mergedSameCols = sameCols + appendList
    print("the mergedSameCols is: {}".format(mergedSameCols))

    singleNodeList = []
    for songleNode in appendList:
        isAppend = False
        for col  in sameCols:
            if is_node_in_col(list(songleNode)[0], col, data.x):
                col.add(list(songleNode)[0])
                isAppend = True
                break
        if not isAppend:
            singleNodeList.append(songleNode)
    sameCols += singleNodeList
    print("the sameCols is: {}".format(sameCols))

    return sameCols


def val_rows(net_row, dataset, criterion, max_iter=100):
    print('---------->>>Start val row')
    for p in net_row.parameters():
        p.requires_grad = False
    net_row.eval()

    data_loader = DataLoader(dataset, batch_size=1)
    val_iter = iter(data_loader)

    # for i in range(max_iter):
    data = val_iter.next()
    preds_row = net_row(data)
    _, preds_row = preds_row.max(1)
    preds_row = preds_row.detach().cpu().numpy()

    edges_graph = data.edge_index.numpy()
    # print(type(preds_row))
    # print(preds_row)
    # print(type(edges_graph))
    # print(edges_graph)

    sameRowMatrix = ouputSameRow(preds_row, edges_graph, data.x)
    sameRows = getClusterSet(sameRowMatrix)
    # print("the sameRows is: {}".format(sameRows))

    appendList = [] # single node
    rowNodes = set()
    for i in sameRows:
        rowNodes = rowNodes | i

    for i in range(len(data.x)):
        if i not in rowNodes:
            appendList.append({i})
    mergedSameRows = sameRows + appendList
    print("the mergedSameRows is: {}".format(mergedSameRows))

    singleNodeList = []
    for songleNode in appendList:
        isAppend = False
        for row  in sameRows:
            if is_node_in_row(list(songleNode)[0], row, data.x):
                row.add(list(songleNode)[0])
                isAppend = True
                break
    if not isAppend:
        singleNodeList.append(songleNode)
    sameRows += singleNodeList
    print("the sameRows is: {}".format(sameRows))
    return sameRows

def is_node_in_row(single_node, row, dataX):
    for node in row:
        if if_not_same_row(single_node, node, dataX):
            return False
    return True

def is_node_in_col(single_node, col, dataX):
    for node in col:
        if if_not_same_col(single_node, node, dataX):
            return False
    return True

def getClusterSet(pairs):
    res = []
    for i in range(len(pairs[1])):
        node1 = pairs[0][i]
        node2 = pairs[1][i]
        if node1==node2:
            continue

        index1 = -1
        index2 = -1
        for j in range(len(res)):
            if node1 in res[j]:
                index1 = j
            if node2 in res[j]:
                index2 = j

        if index1 == index2:
            if index2 == -1:
                newset = {node1,node2}
                res.append(newset)
            else:
                res[index1].add(node1)
                res[index1].add(node2)
        else:
            if index1==-1 or index2==-1:
                index_put = index2 if index1==-1 else index1
                res[index_put].add(node1)
                res[index_put].add(node2)
            else:
                index_keep = min(index1, index2)
                index_remove = max(index1, index2)
                res[index_keep] = res[index_keep].union(res[index_remove])
                res.pop(index_remove)
    print(res)
    return res


def ouputSameCol(preds, edges, dataX):
    indexs = np.where(preds==0)
    pairs = np.delete(edges, indexs, axis=1)
    print("ouputSameCol is {}".format(pairs))
    n = len(pairs[0])
    for i in range(n):
        print("{} - {}".format(pairs[0][i],pairs[1][i]))

    toremoveIndex = []
    for i in range(n):
        if if_not_same_col(pairs[0][i],pairs[1][i], dataX):
            toremoveIndex.append(i)
    pairs = np.delete(pairs, toremoveIndex, axis=1)

    print("-------------------------------")
    for i in range(len(pairs[0])):
        print("{} - {}".format(pairs[0][i],pairs[1][i]))
    return pairs

def ouputSameRow(preds, edges, dataX):
    indexs = np.where(preds==0)
    pairs = np.delete(edges, indexs, axis=1)
    print("ouputSameRow is {}".format(pairs))
    n = len(pairs[0])
    for i in range(n):
        print("{} - {}".format(pairs[0][i],pairs[1][i]))

    toremoveIndex = []
    for i in range(n):
        if if_not_same_row(pairs[0][i],pairs[1][i], dataX):
            toremoveIndex.append(i)
    pairs = np.delete(pairs, toremoveIndex, axis=1)

    print("-------------------------------")
    for i in range(len(pairs[0])):
        print("{} - {}".format(pairs[0][i],pairs[1][i]))
    return pairs

def if_not_same_col(index1,index2,dataX):
    if  index1==index2:
        return True    
    position1 = dataX[index1]
    position2 = dataX[index2]

    if position1[0] > position2[1] or position1[1] < position2[0]:
        return True
    return False


def if_not_same_row(index1,index2,dataX):
    if  index1==index2:
        return True    
    position1 = dataX[index1]
    position2 = dataX[index2]

    if position1[2] > position2[3] or position1[3] < position2[2]:
        return True
    return False

def printData(root_path):
    ds = ValidateDataset(root_path)

    test_loader = DataLoader(ds, batch_size=5 )

    for data in test_loader:
        print(data,data.num_graphs)
        print("data.x:{}".format(data.x))
        # print("ratio:",data.y.sum().float()/data.y.size()[0])
        print(data.imgpos[0:10,:], data.nodenum)
        torch.save(data, "temp.pt")

def form_col_row(cols_set, rows_set, dataX):
    col_count = len(cols_set)
    row_count = len(rows_set)
    res_table = [[True for j in range(col_count)] for i in range(row_count)]

    dataX = dataX.detach().numpy()
    cols_set = sort_col_cells(cols_set, dataX)
    rows_set = sort_row_cells(rows_set, dataX)

    print("-----------")
    for row in rows_set:
        # if len(row) < col_count:
        #     row.insert(0,0)
        print(row)
    print("-----------")
    print(res_table)
    pass



def sort_col_cells(cols_set, dataX):
    def getColEdge(col):
        right_point_list = []
        for node in col:
            right_point_list.append(dataX[node][0])
        return min(right_point_list)

    col_list = copy.deepcopy(cols_set)
    col_list.sort(key=getColEdge)

    newList = []
    for col in col_list:
        newList.append(list(col))
    for col in newList:
        col.sort(key=lambda p: dataX[p][2], reverse=True)
    return newList

def sort_row_cells(rows_set, dataX):
    def getRowEdge(row):
        top_point_list = [dataX[i][2] for i in row]
        return max(top_point_list)

    row_list = copy.deepcopy(rows_set)
    row_list.sort(key=getRowEdge, reverse=True)

    newList = []
    for row in row_list:
        newList.append(list(row))
    for row in newList:
        row.sort(key=lambda p: dataX[p][0], reverse=False)
    return newList

def predicateData(data_path, model_path_col, model_path_row):
    # cols
    test_dataset_col = ValidateDataset(data_path)

    device = torch.device("cpu" )
    model_col = TbNet(input_num,vocab_size,num_text_features,nclass)
    model_col.cpu()
    criterion = torch.nn.NLLLoss()
    model_col.load_state_dict(torch.load(model_path_col))

    cols_set = val_cols(model_col, test_dataset_col, criterion)

    # rows
    test_dataset_row = Validate_RowDataset(data_path)
    model_row = TbNet(input_num,vocab_size,num_text_features,nclass)
    model_row.cpu()
    criterion = torch.nn.NLLLoss()
    model_row.load_state_dict(torch.load(model_path_row))

    rows_set = val_rows(model_row, test_dataset_row, criterion)


    # construct table
    data_loader = DataLoader(test_dataset_col, batch_size=1)
    val_iter = iter(data_loader)
    data = val_iter.next()

    form_col_row(cols_set, rows_set, data.x)

nclass = 2
input_num = 8
vocab_size = 39
num_text_features = 64


data_path = r'C:\Users\zhak\GFTE\testTable\data'
rcnnPath_col = r"C:\Users\zhak\GFTE\expr\net_20_360.pth"
rcnnPath_row = r"C:\Users\zhak\GFTE222\expr\netYY_20_360.pth"

if __name__ == "__main__":  
    # printData(data_path)

    predicateData(data_path, rcnnPath_col, rcnnPath_row)
