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

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz,. "
vob = {x: ind for ind, x in enumerate(alphabet)}


def encode_text(ins, vob, max_len=10, default=" "):
    out = []
    sl = len(ins)
    minl = min(sl, max_len)
    for i in range(minl):
        char = ins[i]
        if char in vob:
            out.append(vob[char])
        else:
            out.append(vob[default])
    if len(out) <= max_len:
        out = out + [vob[default]] * (max_len - len(out))
    return out


class tableDataset(Dataset):
    def __init__(self, root_path, cell_type, transform=None, pre_transform=None):
        super(tableDataset, self).__init__(root_path, cell_type, transform, pre_transform)
        self.root_path = root_path
        self.cell_type = cell_type
        self.transform = transform
        self.pre_transform = pre_transform
        self.jsonfile = os.path.join(self.root_path, "imglist.json")
        self.img_size = 256
        self.kernel = np.ones((3, 3), np.uint8)  # 把图像的线变粗一点
        if os.path.exists(self.jsonfile):  # imglist.json去掉了一些有疑问的文件
            with open(self.jsonfile, "r") as read_file:
                self.imglist = json.load(read_file)
        else:
            self.imglist = list(filter(lambda fn: fn.lower().endswith('.jpg') or fn.lower().endswith('.png'),
                                       os.listdir(os.path.join(self.root_path, "../node-image-test/img"))))
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
        validlist = []
        for idx in range(len(self.imglist)):
            structs, chunks, img, _, _, _ = self.readlabel(idx)
            if structs is None or chunks is None or img is None:
                continue

            vi = self.check_chunks(structs, chunks)
            if vi == 1 and (not img is None):
                validlist.append(self.imglist[idx])
        return validlist

    # structs中不应该有空的cell，但是实际上可能有。空cell存在，会影响在chunk中的index。
    def remove_empty_cell(self, structs):
        structs.sort(key=lambda p: p["id"])
        news = [];
        idx = 0
        for st in structs:
            text = st["tex"].strip().replace(" ", "")
            if text == "" or text == '$\\mathbf{}$':  # 空的cell
                continue
            st["id"] = idx
            news.append(st)
            idx += 1
        return news

    def remove_chunk_empty_cell(self, chunks):
        chunks.sort(key=lambda p: p["id"])
        news = [];
        idx = 0
        for st in chunks:
            text = st["text"].strip().replace(" ", "")
            if text == "" or text == '$\\mathbf{}$':  # 空的cell
                continue
            st["id"] = idx
            news.append(st)
            idx += 1
        return news

    def check_chunks(self, structs, chunks):
        structs = self.remove_empty_cell(structs)
        chunks = self.remove_chunk_empty_cell(chunks)
        for st in structs:
            id = st["id"]
            if id >= len(chunks):
                # print("chunk index out of range.", id)
                return 0
            # ch = chunks[id]
            chk_list = [i for i in chunks if i["id"] == id]
            if len(chk_list) != 1:
                continue
            ch = chk_list[0]

            txt1 = st["tex"].replace(" ", "")
            txt2 = ch["text"].replace(" ", "")
            if txt1 != txt2:
                print(id, "mismatch:", txt1, " ", txt2)
            if st["end_row"] - st["start_row"] != 0 or st["end_col"] - st["start_col"] != 0:
                pass
        return 1

    def format_html(self, structs, chunks):
        rowcnt = max(structs, key=lambda p: p["end_row"])["end_row"] + 1
        colcnt = max(structs, key=lambda p: p["end_col"])["end_col"] + 1
        mat = [["<td></td>"] * colcnt for i in range(rowcnt)]
        for st in structs:  # 填空
            mat[st["start_row"]][st["start_col"]] = "<td>" + st["tex"] + "</td>"
        html = ""
        # print(mat)
        for row in mat:
            html += "<tr>" + "".join(row) + "</tr>"
        return html

    def readlabel(self, idx):
        imgfn = self.imglist[idx]
        #         print('img: {}'.format(imgfn))
        structfn = os.path.join(self.root_path, "structure", os.path.splitext(os.path.basename(imgfn))[0] + ".json")
        chunkfn = os.path.join(self.root_path, "../node-image-test/chunk", os.path.splitext(os.path.basename(imgfn))[0] + ".chunk")
        imgfn = os.path.join(self.root_path, "../node-image-test/img", os.path.splitext(os.path.basename(imgfn))[0] + ".png")
        segLine_fn = os.path.join(self.root_path, "seg_label", os.path.splitext(os.path.basename(imgfn))[0] + ".json")

        if not os.path.exists(structfn) or not os.path.exists(chunkfn) or not os.path.exists(
                imgfn) or not os.path.exists(segLine_fn):
            if not os.path.exists(structfn):
                print("can't find files structfn.")
            if not os.path.exists(chunkfn):
                print("can't find files chunkfn.")
            if not os.path.exists(imgfn):
                print("can't find files imgfn.")
            if not os.path.exists(segLine_fn):
                print("can't find files segLine_fn.")
            return None, None, None, None, None
        # with open(chunkfn, 'r') as f:
        #     chunks = json.load(f)['chunks']
        chunks = json.load(codecs.open(chunkfn, 'r', 'utf-8-sig'))['chunks']
        if len(chunks) == 0:
            print(chunkfn)
        # with open(structfn, 'r') as f:
        #     structs = json.load(f)['cells']
        structs = json.load(codecs.open(structfn, 'r', 'utf-8-sig'))['cells']

        # get segment lines
        with open(segLine_fn, 'r') as f:
            segLines = json.load(f)

        img = cv2.imread(imgfn)
        if not img is None:
            height = img.shape[0]
            width = img.shape[1]
            seg_header = (1 - segLines["y_header"]) * height
            seg_att = segLines["x_attribute"] * width

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = 255 - img
            img = cv2.dilate(img, self.kernel, iterations=1)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        return structs, chunks, img, seg_header, seg_att, imgfn

    def __len__(self):
        return len(self.imglist)

    def box_center(self, chkp):
        # x1, x2, y1, y2  in chkp
        return [(chkp[0] + chkp[1]) / 2, (chkp[2] + chkp[3]) / 2]

    def get_html(self, idx):
        structs, chunks, _, _, _ = self.readlabel(idx)
        self.check_chunks(structs, chunks)
        html = self.format_html(structs, chunks)
        return html

    def cal_chk_limits(self, chunks):
        x_min = min(chunks, key=lambda p: p["pos"][0])["pos"][0]
        x_max = max(chunks, key=lambda p: p["pos"][1])["pos"][1]
        y_min = min(chunks, key=lambda p: p["pos"][2])["pos"][2]
        y_max = max(chunks, key=lambda p: p["pos"][3])["pos"][3]
        hlist = [p["pos"][3] - p["pos"][2] for p in chunks]
        avhei = sum(hlist) / len(hlist)
        # 加入一点边界, 大概对应整个图像。
        width = x_max - x_min + 2 * avhei
        height = y_max - y_min + 0.5 * 2 * avhei
        return [x_min, x_max, y_min, y_max, width, height, avhei]  #

    # 相对的位置。
    def pos_feature(self, chk, cl):
        x1 = (chk["pos"][0] - cl[0] + cl[6]) / cl[4]
        x2 = (chk["pos"][1] - cl[0] + cl[6]) / cl[4]
        x3 = (chk["pos"][2] - cl[2] + 0.5 * cl[6]) / cl[5]
        x4 = (chk["pos"][3] - cl[2] + 0.5 * cl[6]) / cl[5]
        x5 = (x1 + x2) * 0.5  # 中心点
        x6 = (x3 + x4) * 0.5
        x7 = x2 - x1  # 文本宽度
        x8 = x4 - x3  # 文本高度
        return [x1, x2, x3, x4, x5, x6, x7, x8]

    def augmentation_chk(self, chunks):
        for chk in chunks:
            chk["pos"][0] += random.normalvariate(0, 1)
            chk["pos"][1] += random.normalvariate(0, 1)
            chk["pos"][2] += random.normalvariate(0, 1)
            chk["pos"][3] += random.normalvariate(0, 1)

    def get(self, idx):
        structs, chunks, img, header_line, att_line, imgfn = self.readlabel(idx)

        # imgfn = self.imglist[idx]
        # print("now is {}".format(imgfn))

        structs = self.remove_empty_cell(structs)
        chunks = self.remove_chunk_empty_cell(chunks)

        cl = self.cal_chk_limits(chunks)
        # header_line = (1 - segLines["y_header"])*cl[6]
        # att_line = segLines["x_attribute"]*cl[5]

        # select cells in cell_type
        cell_nodes = []
        # header
        if (self.cell_type == 0):
            for cell in chunks:
                x_left = cell['pos'][0]
                x_right = cell['pos'][1]
                y_bottom = cell['pos'][2]  # y的值从大到小排，【2】是ymin，是底部的值
                y_top = cell['pos'][3]
                x_mid = (x_left + x_right) * 0.5
                y_mid = (y_top + y_bottom) * 0.5
                if x_mid >= att_line and y_mid >= header_line:
                    cell_nodes.append(cell)
        # attributer
        if (self.cell_type == 1):
            for cell in chunks:
                x_left = cell['pos'][0]
                x_right = cell['pos'][1]
                y_bottom = cell['pos'][2]  # y的值从大到小排，【2】是ymin，是底部的值
                y_top = cell['pos'][3]
                x_mid = (x_left + x_right) * 0.5
                y_mid = (y_top + y_bottom) * 0.5
                if x_mid <= att_line and y_mid <= header_line:
                    cell_nodes.append(cell)
        # data
        if (self.cell_type == 2):
            for cell in chunks:
                x_left = cell['pos'][0]
                x_right = cell['pos'][1]
                y_bottom = cell['pos'][2]  # y的值从大到小排，【2】是ymin，是底部的值
                y_top = cell['pos'][3]
                x_mid = (x_left + x_right) * 0.5
                y_mid = (y_top + y_bottom) * 0.5
                if x_mid >= att_line and y_mid <= header_line:
                    cell_nodes.append(cell)
        # print("this header has {} cells.".format(len(header_nodes)))

        # print(cl)
        # x = [chunks[st["id"]]["pos"] for st in structs]
        x, pos, tbpos, xtext, imgpos = [], [], [], [], []
        plaintext = []

        for st in structs:
            id = st["id"]
            # chk = chunks[id]
            chk_list = [i for i in cell_nodes if i["id"] == id]
            if len(chk_list) != 1:
                continue
            chk = chk_list[0]

            xt = self.pos_feature(chk, cl)
            x.append(xt)
            pos.append(xt[4:6])
            tbpos.append([st["start_row"], st["end_row"], st["start_col"], st["end_col"]])
            xtext.append(encode_text(chk["text"], vob))
            plaintext.append(chk["text"].encode('utf-8'))
            imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])  # 图像中的y是倒过来的。这是归一化[-1,1]之间。图像的y在前，和H对应。

        # 获取id
        id_list = []
        for i in range(len(cell_nodes)):
            id_list.append(cell_nodes[i]["id"])

        x = torch.FloatTensor(x)
        pos = torch.FloatTensor(pos)
        data = Data(x=x, pos=pos)
        # data = self.graph_transform(data) # 构造图的连接

        graph_transform = GT.KNNGraph(k=len(cell_nodes))
        data = graph_transform(data)
        # print("在数据集中data_edge_index的数目为:{}".format(data.edge_index.size()[1]))

        y = self.cal_label(data, tbpos)
        img = torch.FloatTensor(img / 255.0).unsqueeze(0).unsqueeze(0)
        # print(img.size(), img.dtype)
        data.y = torch.LongTensor(y)
        data.img = img
        data.imgpos = torch.FloatTensor(imgpos)
        data.nodenum = torch.LongTensor([len(cell_nodes)])
        # print(type(xtext)) #<class 'list'>
        data.xtext = torch.LongTensor(xtext)
        data.id_list = torch.LongTensor(id_list)
        # print(type(plaintext))
        # print(plaintext)
        data.plaintext = plaintext
        data.tbpos = torch.LongTensor(tbpos)
        data.imgfn = imgfn.encode('utf-8')
        # print(data)
        return data

    def cal_label(self, data, tbpos):  # 根据构造的图，计算边的标注。
        edges = data.edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y = []
        for i in range(edges.size()[1]):
            y.append(self.get_col_row(edges[0, i], edges[1, i], tbpos))
        return y

    def get_col_row(self, si, ti, tbpos):
        if self.if_same_row(si, ti, tbpos) == 1:
            return 1
        if self.if_same_col(si, ti, tbpos) == 1:
            return 2
        else:
            return 0

    def if_same_row(self, si, ti, tbpos):
        ss, se = tbpos[si][0], tbpos[si][1]
        ts, te = tbpos[ti][0], tbpos[ti][1]
        if (ss >= ts and se <= te):
            return 1
        if (ts >= ss and te <= se):
            return 1
        return 0

    def if_same_col(self, si, ti, tbpos):
        ss, se = tbpos[si][2], tbpos[si][3]
        ts, te = tbpos[ti][2], tbpos[ti][3]
        if (ss >= ts and se <= te):
            return 1
        if (ts >= ss and te <= se):
            return 1
        return 0
