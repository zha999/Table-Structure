#!/usr/bin/env python
# coding: utf-8

# 根据gfte预测结果构造两个邻接矩阵，用辅助信息修正
# 对header利用最大团算法找出同行的节点，对各个极大团排序，确定行数
# 对header利用最大团算法找出同列的节点，对各个极大团排序，确定列数
# 对attributer利用最大团算法找出同行的节点，对各个极大团排序，确定行数
# 对attributer利用最大团算法找出同列的节点，对各个极大团排序，确定列数
# 构建二维网格，填充data
# 处理corner

# from skimage.io import imread
# from skimage.util import crop
# from skimage.transform import rotate,resize,rescale
import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon
# from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
import csv
import codecs
import numpy as np

from post_processor.insertData import insert_data
from post_processor.model import TbNet
from post_processor.predsDataset import predsDataset
from post_processor.tableDataset import tableDataset


# In[89]:


def readst(structfn):
    correct_label = []
    structs = json.load(codecs.open(structfn, 'r', 'utf-8-sig'))['cells']
    for st in structs:
        id = st["id"] 
        tmp = node(id, st["tex"], 0, [], st["start_row"], st["end_row"], st["start_col"], st["end_col"])
        correct_label.append(tmp)
    
    return correct_label

# In[93]:


# 用于表格重建
# id为在chunk/structure中的id
# pos = [xmin,xmax,ymin,ymax,xmid,ymid]
# node_type:
# header:0 attributer:1 data:2 corner:3

class node:
    
    def __init__(self, id, tex, node_type, pos = [], start_row=0, end_row=0, start_col=0, end_col=0):
        
        self.id = id
        self.tex = tex
        self.node_type = node_type
        self.start_row = start_row
        self.end_row = end_row
        self.start_col = start_col
        self.end_col = end_col
        self.pos = pos


# In[94]:

# preds = preds.detach().cpu().numpy()
# label = data.y.detach().cpu().numpy()
# preds类型为numpy.ndarray,元素个数与data的edge边数相同

'''
测试时batchsize均取1
data.nodenum
data.x[j]
data.plaintext[0][j].decode('utf-8')
data.edge_index的格式为[2][该batch所有表格的边的拼接]
'''

'''
construct_matrix使用时
h_row_matrix, h_col_matrix = construct_matrix(preds_header, data_header)
'''



# index为该batch的第index个表格
# start_edge为正在处理的表格的边在data.edge_index中的起始位置
# start_node为正在处理的表格的节点在data.x中的起始位置
# part_id为正在处理的部分的id列表

def construct_matrix(preds, data):
    # 构造邻接矩阵，设节点数目为n，则矩阵规模为n*n
    # data.nodenum为节点数目
    nodenum = data.nodenum
    edge_num = nodenum*(nodenum-1) # 完全图
    edges = data.edge_index
    
    # data.x是节点的位置信息，[xmin,xmax,ymin,ymax,xmid,ymid,width,height]
    x = data.x
    row_matrix = np.zeros((nodenum, nodenum),dtype=np.int)
    col_matrix = np.zeros((nodenum, nodenum),dtype=np.int)
    
    
    for i in range(len(preds)):
        s_node = edges[0][i]
        e_node = edges[1][i]
    
        #如果垂直方向有重叠则判断为同列    
        if(preds[i]!=2 and (((x[s_node][0]>x[e_node][0] or x[s_node][0]>=x[e_node][0]) and (x[s_node][0]<x[e_node][1] or x[s_node][0]<=x[e_node][1])) or 
           ((x[s_node][0]<x[e_node][0] or x[s_node][0]<=x[e_node][0]) and (x[s_node][1]>x[e_node][0] or x[s_node][1]>=x[e_node][0])))):

            preds[i] = 2
            
        if(preds[i]!=2 and (((x[e_node][0]>x[s_node][0] or x[e_node][0]>=x[s_node][0]) and (x[e_node][0]<x[s_node][1] or x[e_node][0]<=x[s_node][1])) or 
           ((x[e_node][0]<x[s_node][0] or x[e_node][0]<=x[s_node][0]) and (x[e_node][1]>x[s_node][0] or x[e_node][1]>=x[s_node][0])))):

            preds[i] = 2
            # print("i:{},2修改了".format(i))

        # e_node的中心点x坐标在s_node的xmin～xmax之间
        if(preds[i]!=2 and (x[s_node][0]<x[e_node][4] and x[s_node][1]>x[e_node][4])):

            preds[i] = 2
            # print("3修改了")

        # s_node的中心点x坐标在e_node的xmin～xmax之间
        if(preds[i]!=2 and (x[e_node][0]<x[s_node][4] and x[e_node][1]>x[s_node][4])):

            preds[i] = 2
            # print("4修改了")

        
        # e_node的中心点y坐标在s_node的ymin～ymax之间
        if(preds[i]!=1 and (x[s_node][2]<x[e_node][5] and x[s_node][3]>x[e_node][5])):
            
            preds[i] = 1
            # print("7修改了")
            
        # s_node的中心点y坐标在e_node的ymin～ymax之间
        if(preds[i]!=1 and (x[e_node][2]<x[s_node][5] and x[e_node][3]>x[s_node][5])):
            
            preds[i] = 1
            # print("8修改了")
    
    
    
    for i in range(len(preds)):
        s_node = edges[0][i]
        e_node = edges[1][i]
        if(row_matrix[s_node][e_node]==0):
            row_matrix[s_node][e_node] = 1 if preds[i]==1 else 0
            row_matrix[e_node][s_node] = 1 if preds[i]==1 else 0
        if(col_matrix[s_node][e_node]==0):
            col_matrix[s_node][e_node] = 1 if preds[i]==2 else 0
            col_matrix[e_node][s_node] = 1 if preds[i]==2 else 0
        #print("s_node:{}, e_node:{}, col_matrix[s_node][e_node]:{}".format(s_node,e_node,col_matrix[s_node][e_node]))
        
                
    return row_matrix, col_matrix, preds


# In[95]:


# 极大团算法

'''

# 使用方法
    cliques = []
    maxn = 50;
    # 分别是P集合，X集合，R集合
    some = np.zeros((maxn,maxn),dtype = np.int)
    none = np.zeros((maxn,maxn),dtype = np.int)
    r = np.zeros((maxn,maxn),dtype = np.int)
    for i in range(data.nodenum):
        some[1][i] = i;
        i += 1
    adj_matrix = r_new（待处理邻接矩阵）
    ans = 0
    dfs(1, 0, data.nodenum, 0)
    
    结果示例：
    ans = 5
    cliques:[array([0, 1]), array([2, 3, 4, 5]), array([ 6,  7,  8,  9, 10]), array([11, 12, 13, 14, 15]), array([16, 17, 18, 19])]

'''

import numpy as np
from copy import deepcopy

    
def dfs(d, an, sn, nn, cliques, some, none, r, adj_matrix):

    if(sn==0 and nn==0):
        pass
    u = some[d][0];  # 选取Pivot结点
    for i in range(sn):
        v = some[d][i]
        
        if(adj_matrix[u][v]==1):
            continue
        # 如果是邻居结点，就直接跳过下面的程序，进行下一轮的循环。显然能让程序运行下去的，只有两种，一种是v就是u结点本身，另一种是v不是u的邻居结点。
        for j in range(an):
            r[d+1][j] = r[d][j]
        
        r[d+1][an] = v
        # 用来分别记录下一层中P集合和X集合中结点的个数
        tsn = 0
        tnn = 0
        for j in range(sn):
            if(adj_matrix[v][some[d][j]]):
                
                some[d+1][tsn] = some[d][j]
                tsn += 1
            
        for j in range(nn):
            if(adj_matrix[v][none[d][j]]): 
                none[d+1][tnn] = none[d][j]
                tnn += 1
            
        if(tsn==0 and tnn==0):
            tmp_r = deepcopy(r[d+1])
            tmp_r = np.unique(tmp_r)
            tmp_r = tmp_r[tmp_r != 0]
            cliques.append(deepcopy(tmp_r))
            
        dfs(d+1, an+1, tsn, tnn, cliques, some, none, r, adj_matrix)
        
        some[d][i] = 0
        
        none[d][nn] = v
        nn += 1


# In[96]:


# 根据极大团的个数分配行数和列数
# 行数根据row_matrix得到的极大团确定
# 列数根据col_matrix得到的极大团确定

# 快速排序
def partition(arr,cliques,low,high): 
    i = ( low-1 )         # 最小元素索引
    pivot = arr[high]
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot 
        if   arr[j] <= pivot: 
          
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]
            cliques[i],cliques[j] = cliques[j],cliques[i]
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    cliques[i+1],cliques[high] = cliques[high],cliques[i+1]
    return ( i+1 ) 
  
# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引
  
# 快速排序函数
def quickSort(arr,cliques,low,high): 
    if low < high: 
  
        pi = partition(arr,cliques,low,high) 
  
        quickSort(arr, cliques, low, pi-1) 
        quickSort(arr, cliques, pi+1, high) 
''' 
使用示例
arr = [10, 7, 8, 9, 1] 
cliques = [np.array([0, 1]), np.array([2, 3, 4, 5]), np.array([ 6,  7,  8,  9, 10]), np.array([11, 12, 13, 14, 15]), np.array([16, 17, 18, 19])]
n = len(arr) 
quickSort(arr,cliques,0,n-1)

print ("排序后的数组:{}".format(arr)) 
print("排序后的极大团:{}".format(cliques))

'''

# 确定行数/列数需要节点的位置信息
# 如果确定行数，那么返回行数最大的节点的id列表，用于插入data部分
# 如果确定列数，那么返回列数最大的节点的id列表，用于插入data部分

def r_c_number(cliques, data, row=True, col=False):
    
    #计算极大团列表cliques中x/y坐标的平均值
    
    if(row): #根据y坐标确定行数
        nodenum = data.nodenum
        x = data.x
        avg_pos = []
        for i in range(len(cliques)):
            sum_pos = 0.0
            for j in range(len(cliques[i])):
                #print(j)
                sum_pos += x[cliques[i][j]][5]
            avg_pos.append(deepcopy(sum_pos/len(cliques[i])))
            
        #根据avg_pos对cliques排序
        quickSort(avg_pos, cliques, 0, len(cliques)-1)
        
        start_row = [0]*nodenum # start_row
        end_row = [0]*nodenum # end_row
        bool_span = [0]*nodenum # 用来判断该node是否存在于多个极大团中
        number = 0 # 当前行数
        row_max = [] # 行数最大的node
        for i in range(len(cliques)):
            i = len(cliques) - i - 1 # 以左下角为原点，则y越大，行数越小
            nodes = cliques[i]
            for j in range(len(nodes)):
                node_index = nodes[j]
                if(i == 0):
                    row_max.append(nodes[j])
                if(bool_span[node_index]==0):
                    bool_span[node_index]=1
                    start_row[node_index]=number
                    end_row[node_index]=number
                else:
                    #如果新的行数小于现在的start_row，则更新
                    if(number < start_row[node_index]):
                        start_row[node_index] = number
                    #如果新的行数大于现在的end_row，则更新
                    if(number > end_row[node_index]):
                        end_row[node_index] = number
            number += 1
    if(col): #根据x坐标确定列数
        nodenum = data.nodenum
        x = data.x
        avg_pos = []
        for i in range(len(cliques)):
            sum_pos = 0.0
            for j in range(len(cliques[i])):
                sum_pos += x[cliques[i][j]][4]
            avg_pos.append(deepcopy(sum_pos/len(cliques[i])))
        
        #根据avg_pos对cliques排序
        quickSort(avg_pos,cliques,0,len(cliques)-1)
        
        start_col = [0]*nodenum # start_col
        end_col = [0]*nodenum # end_col
        bool_span = [0]*nodenum # 用来判断该node是否存在于多个极大团中
        col_max = [] # 列数最大的node的id
        number = 0 # 当前列数
        # print("计算列数的cliques:{}".format(cliques))
        for i in range(len(cliques)):
            nodes = cliques[i]

            for j in range(len(nodes)):
                node_index = nodes[j]
                if(i==len(cliques)-1):
                    col_max.append(nodes[j])
                if(bool_span[node_index]==0):
                    bool_span[node_index]=1
                    start_col[node_index]=number
                    end_col[node_index]=number
                else:
                    #如果新的行数小于现在的start_col，则更新
                    if(number < start_col[node_index]):
                        start_col[node_index] = number
                    #如果新的行数大于现在的end_col，则更新
                    if(number > end_col[node_index]):
                        end_col[node_index] = number
            number += 1
            
    if(row):       
        return start_row, end_row, row_max
    if(col):       
        return start_col, end_col, col_max

# In[98]:


def wirtejson(node_list, root_path, name):
    json_content = {}
    cells = []
    for i in range(len(node_list)):
        cell = {}
        cell['id'] = node_list[i].id
        cell['tex'] = node_list[i].tex
        cell['start_row'] = node_list[i].start_row
        cell['end_row'] = node_list[i].end_row
        cell['start_col'] = node_list[i].start_col
        cell['end_col'] = node_list[i].end_col
        cells.append(cell)
    
    json_content['cells'] = cells
    json_file_path = os.path.join(root_path, name+".json")
    with open(json_file_path, 'w') as f:
        json.dump(json_content, f) # , ensure_ascii=False
    f.close()

def format_html(structs):
        rowcnt = max(structs, key=lambda p: p["end_row"])["end_row"]+1
        colcnt = max(structs, key=lambda p: p["end_col"])["end_col"]+1
        # print("row , col number:", rowcnt, colcnt)
        mat = [["<td></td>"]*colcnt for i in range(rowcnt)]
        for st in structs: # 填空
            mat[st["start_row"]][st["start_col"]] = "<td>" + st["tex"] + "</td>"
        html = ""
        #print(mat)
        for row in mat:
            html += "<tr>"+"".join(row)+"</tr>"
        return html


# In[99]:


def cal_get_col_row(si, ti, part_list):
        if(cal_if_same_row(si, ti, part_list) == 1):
            return 1
        if(cal_if_same_col(si, ti, part_list) == 1):
            return 2
        else:
            return 0
            
def cal_if_same_row(si,ti,part_list):
    ss,se = part_list[si].start_row, part_list[si].end_row
    ts,te = part_list[ti].start_row, part_list[ti].end_row
    if (ss>=ts and se<=te):
        return 1
    if (ts>=ss and te<=se):
        return 1
    return 0
    
def cal_if_same_col(si,ti,part_list):
    ss,se = part_list[si].start_col, part_list[si].end_col
    ts,te = part_list[ti].start_col, part_list[ti].end_col
    if (ss>=ts and se<=te):
        return 1
    if (ts>=ss and te<=se):
        return 1
    return 0


def cal_cal_label(data, part_list): # 根据构造的图，计算边的标注。
    edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
    y = []
    for i in range(edges.size()[1]):
        # y.append(self.if_same_row(edges[0,i], edges[1,i],tbpos))
        # y.append(self.if_same_col(edges[0,i], edges[1,i],tbpos))
        y.append(cal_get_col_row(edges[0, i], edges[1, i], part_list))
    return y


# In[100]:


def construct_html(root_path, table_node, imgfn):
    table = HTMLTable(caption=imgfn)
    row_sum = 0
    col_sum = 0
    wait_merge_row = []
    wait_merge_col = []
    
    for k in range(len(table_node)):
        if(table_node[k].end_row>row_sum):
            row_sum = table_node[k].end_row
        if(table_node[k].end_col>col_sum):
            col_sum = table_node[k].end_col
    # print("row_sum:{}, col_sum:{}".format(row_sum, col_sum))
    
    row_sum += 1
    col_sum += 1
    
    table_html = []
    for k in range(row_sum):
        tmp_row = []
        for j in range(col_sum):
            tmp_col = ['']
            tmp_row.append(tmp_col)
        table_html.append(tmp_row)
    
    # print("table_html:{}".format(table_html))
    
    for k in range(len(table_node)):
        start_row = table_node[k].start_row
        end_row = table_node[k].end_row
        start_col = table_node[k].start_col
        end_col = table_node[k].end_col
        if(start_row!=end_row):
            wait_merge_row.append(table_node[k])
        if(start_col!=end_col):
            wait_merge_col.append(table_node[k])
        
        for p in range(end_row-start_row+1):
            for q in range(end_col-start_col+1):
                # print("start_row+p:{},start_col+q:{}".format(start_row+p,start_col+q))
                table_html[start_row+p][start_col+q] = table_node[k].tex
    
    table.append_data_rows(table_html)
    
    for k in range(len(wait_merge_row)):
        index_row = wait_merge_row[k].start_row
        index_col = wait_merge_row[k].start_col
        span_num = wait_merge_row[k].end_row - wait_merge_row[k].start_row + 1
        table[index_row][index_col].attr.rowspan = span_num
    
    for k in range(len(wait_merge_col)):
        index_row = wait_merge_col[k].start_row
        index_col = wait_merge_col[k].start_col
        span_num = wait_merge_col[k].end_col - wait_merge_col[k].start_col + 1
        table[index_row][index_col].attr.colspan = span_num
        
    # 标题样式
    table.caption.set_style({
        'font-size': '15px',
    })

    # 表格样式，即<table>标签样式
    table.set_style({
        'border-collapse': 'collapse',
        'word-break': 'keep-all',
        'white-space': 'nowrap',
        'font-size': '14px',
    })

    # 统一设置所有单元格样式，<td>或<th>
    table.set_cell_style({
        'border-color': '#000',
        'border-width': '1px',
        'border-style': 'solid',
        'padding': '5px',
    })

    html = table.to_html()
    with open(os.path.join(root_path, 'table_html', imgfn +'.html'), 'w', encoding="utf-8-sig") as f:
            f.write(html)
    f.close()

# In[103]:


# 输入：待计算区域的节点数目，由preds计算得到的矩阵，计算行序号/计算列序号
# 输出：start_row/col, end_row/col, a/h_row/col_max_id
def calculate_row_col(area_data, graph, row=True, col=False):
    
    if(area_data.nodenum!=1):
        # 找出行极大团
        cliques = []
        maxn = 100

        # 分别是P集合，X集合，R集合
        some = np.zeros((maxn,maxn), dtype = np.int)
        none = np.zeros((maxn,maxn), dtype = np.int)
        r = np.zeros((maxn,maxn), dtype = np.int)

        for j in range(area_data.nodenum):
            some[1][j] = j+1

        adj_matrix = np.zeros((area_data.nodenum+1,area_data.nodenum+1), dtype = np.int)
        for p in range(area_data.nodenum):
            for q in range(area_data.nodenum):
                if(graph[p][q]==1):
                    adj_matrix[p+1][q+1] = 1

        dfs(1, 0, area_data.nodenum, 0, cliques, some, none, r, adj_matrix)
        for j in range(len(cliques)):
            for k in range(len(cliques[j])):
                cliques[j][k] -= 1

        # 根据极大团确定行数
        if(row):
            start, end, max_id = r_c_number(cliques, area_data, row=True, col=False)
        elif(col):
            start, end, max_id = r_c_number(cliques, area_data, row=False, col=True)
            
    else:
        start = [0]
        end = [0] 
        max_id = [0]
        
    return start, end, max_id


# In[104]:


from sklearn.metrics import confusion_matrix
from HTMLTable import HTMLTable

def post_proceed(root_path):

    header_type = 0
    attr_type = 1
    data_type = 2

    # header_ds_test = tableDataset(root_path, header_type)
    header_ds_test = predsDataset(root_path, header_type)
    attributer_ds_test = predsDataset(root_path, attr_type)
    t_data_ds_test = predsDataset(root_path, data_type)
    
    nclass = 3
    input_num = 8
    vocab_size = 39
    num_text_features = 64
    device = torch.device("cpu" )
    h_model = TbNet(input_num, vocab_size, num_text_features,nclass) #.cuda()
    h_pthfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "net_50_125.pth")
    h_model.load_state_dict(torch.load(h_pthfile, map_location=torch.device('cpu')))

    a_model = TbNet(input_num, vocab_size, num_text_features,nclass) #.cuda()
    a_pthfile = os.path.join(os.path.split(os.path.abspath(__file__))[0], "net_50_102.pth") # './net_50_102.pth'
    a_model.load_state_dict(torch.load(a_pthfile, map_location=torch.device('cpu')))

    header_loader = DataLoader(header_ds_test, batch_size = 1)
    header_iter = iter(header_loader)

    attributer_loader = DataLoader(attributer_ds_test, batch_size=1)
    attributer_iter = iter(attributer_loader)
    
    t_data_loader = DataLoader(t_data_ds_test, batch_size=1)
    t_data_iter = iter(t_data_loader)

    header_loader_length = len(header_loader)
    for i in range(header_loader_length):
        print(len(header_loader))

        print("这是第{}个表格".format(i))
    
        t_in_ds = i

        header = []
        attributer = []
        t_data_list = []
        corner = []

        '''
        ***************处理header***************
        '''

        h_data = header_iter.next()
        print(h_data.imgfn)
        if(h_data.nodenum!=1):
            h_preds = h_model(h_data)
            _, h_preds = h_preds.max(1)
            h_preds = h_preds.detach().cpu().numpy()
            # h_label = h_data.y.detach().cpu().numpy()


            # 建立邻接矩阵
            # h_row_m, h_col_m, h_preds = construct_matrix(h_label, h_data)
            h_row_m, h_col_m, h_preds = construct_matrix(h_preds, h_data)
            

        else:
            h_row_m, h_col_m, h_preds = None, None, None
        
        start_row, end_row, h_row_max_id = calculate_row_col(h_data, h_row_m, True, False)
        start_col, end_col, h_col_max_id = calculate_row_col(h_data, h_col_m, False, True)

        for j in range(len(h_data.id_list)):
            tmp_pos = h_data.x[j][0:6]
            tmp = node(h_data.id_list[j].item(), h_data.plaintext[0][j].decode('utf-8'), 0, pos = tmp_pos.tolist())
            header.append(tmp)

            h_edges = h_data.edge_index
        
        h_row_max = []
        h_col_max = []


        for j in range(len(header)):
            header[j].start_row = start_row[j]
            header[j].end_row = end_row[j]
        for j in range(len(header)):
            header[j].start_col = start_col[j]
            header[j].end_col = end_col[j]
        for j in h_row_max_id:
            h_row_max.append(header[j])
        for i in h_col_max_id:
            h_col_max.append(header[j])


        '''
        ***************处理attributer***************
        '''


        a_data = attributer_iter.next()
        if(a_data.nodenum != 1):
            a_preds = a_model(a_data)
            _, a_preds = a_preds.max(1)
            a_preds = a_preds.detach().cpu().numpy()
            # a_label = a_data.y.detach().cpu().numpy()
        else:
            a_preds = np.array([0])
            # a_label = np.array([0])

        for j in range(len(a_data.id_list)):
                tmp_pos = a_data.x[j][0:6]
                tmp = node(a_data.id_list[j].item(), a_data.plaintext[0][j].decode('utf-8'), 1, pos = tmp_pos.tolist())
                attributer.append(tmp)

        # 建立邻接矩阵
        if(a_data.nodenum != 1):
            a_row_m, a_col_m, a_preds = construct_matrix(a_preds, a_data)
            # a_row_m, a_col_m, a_preds = construct_matrix(a_label, a_data)
        else:
            a_row_m, a_col_m, a_preds = None, None, None
        
        start_row, end_row, a_row_max_id = calculate_row_col(a_data, a_row_m, True, False)
        start_col, end_col, a_col_max_id = calculate_row_col(a_data, a_col_m, False, True)

        a_row_max = []
        a_col_max = []

        for j in range(len(attributer)):
            attributer[j].start_row = start_row[j]
            #print("start_row[j]:{}".format(attributer[j].start_row))
            attributer[j].end_row = end_row[j]
            #print("end_row[j]:{}".format(attributer[j].start_row))
        for j in range(len(attributer)):
            attributer[j].start_col = start_col[j]
            attributer[j].end_col = end_col[j]
        for j in a_row_max_id:
            a_row_max.append(attributer[j])
        for j in a_col_max_id:
            a_col_max.append(attributer[j])


        '''
        ***************处理t_data***************
        '''
        t_data = t_data_iter.next()
        #t_preds = model(t_data)
        #_, t_preds = t_preds.max(1)
        #t_preds = t_preds.detach().cpu().numpy()
        # t_label = t_data.y.detach().cpu().numpy()
        for j in range(len(t_data.id_list)):
            tmp_pos = t_data.x[j][0:6]
            tmp = node(t_data.id_list[j].item(), t_data.plaintext[0][j].decode('utf-8'), 3, pos = tmp_pos.tolist())
            t_data_list.append(tmp)

        header, attributer, t_data_list = insert_data(t_data_list, header, attributer, h_row_max, a_col_max)


        # 记录各区域ID，将header，attributer，t_data_list按照id排序
        id_h = []
        id_a = []
        id_t_d = []

        # 将不同区域合并
        row_h_sum = 0
        for k in range(len(header)):
            if(header[k].end_row>row_h_sum):
                row_h_sum = header[k].end_row
        row_h_sum += 1 #因为行数的编号是从0开始的，这里需要的总行数

        col_a_sum = 0
        for k in range(len(attributer)):
            if(attributer[k].end_col>col_a_sum):
                col_a_sum = attributer[k].end_col
        col_a_sum += 1

        table_node = []

        for k in range(len(attributer)):
            attributer[k].start_row += row_h_sum
            attributer[k].end_row += row_h_sum
            id_a.append(attributer[k].id)
            table_node.append(attributer[k])

        for k in range(len(header)):
            header[k].start_col += col_a_sum
            header[k].end_col += col_a_sum
            id_h.append(header[k].id)
            table_node.append(header[k])

        for k in range(len(t_data_list)):
            t_data_list[k].start_row += row_h_sum
            t_data_list[k].end_row += row_h_sum
            t_data_list[k].start_col += col_a_sum
            t_data_list[k].end_col += col_a_sum
            id_t_d.append(t_data_list[k].id)
            table_node.append(t_data_list[k])

        json_file_name = os.path.basename(t_data.imgfn[0].decode('utf-8-sig')).split('.')[0]
        wirtejson(table_node, os.path.join(root_path,'pred_structure'), json_file_name)
        construct_html(root_path, table_node, json_file_name)
        
        print("已完成")


# root_path = '../Test'
# post_proceed(root_path)
