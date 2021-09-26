# 将header部分按位置关系分行后进行preds修正

import numpy as np
from copy import deepcopy


def dfs(d, an, sn, nn, cliques, some, none, r, adj_matrix):
    if (sn == 0 and nn == 0):
        pass
    u = some[d][0];  # 选取Pivot结点
    for i in range(sn):
        v = some[d][i]

        if (adj_matrix[u][v] == 1):
            continue
        # 如果是邻居结点，就直接跳过下面的程序，进行下一轮的循环。显然能让程序运行下去的，只有两种，一种是v就是u结点本身，另一种是v不是u的邻居结点。
        for j in range(an):
            r[d + 1][j] = r[d][j]

        r[d + 1][an] = v
        # 用来分别记录下一层中P集合和X集合中结点的个数
        tsn = 0
        tnn = 0
        for j in range(sn):
            if (adj_matrix[v][some[d][j]]):
                some[d + 1][tsn] = some[d][j]
                tsn += 1

        for j in range(nn):
            if (adj_matrix[v][none[d][j]]):
                none[d + 1][tnn] = none[d][j]
                tnn += 1

        if (tsn == 0 and tnn == 0):
            tmp_r = deepcopy(r[d + 1])
            tmp_r = np.unique(tmp_r)
            tmp_r = tmp_r[tmp_r != 0]
            cliques.append(deepcopy(tmp_r))

        dfs(d + 1, an + 1, tsn, tnn, cliques, some, none, r, adj_matrix)

        some[d][i] = 0

        none[d][nn] = v
        nn += 1


# 快速排序只处理列表版, 从大到小排序
def high_to_low_partition(avg_pos, arr, low, high):
    i = low - 1  # 最小元素索引
    pivot = high

    for j in range(low, high):

        # 当前元素小于或等于 pivot
        if avg_pos[j] >= avg_pos[pivot]:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return (i + 1)


# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引

# 快速排序函数
def high_to_low_quickSort(avg_pos, arr, low, high):
    if low < high:
        pi = high_to_low_partition(avg_pos, arr, low, high)

        high_to_low_quickSort(avg_pos, arr, low, pi - 1)
        high_to_low_quickSort(avg_pos, arr, pi + 1, high)


def if_correct_same_col(node1, node2):
    if (node1.pos[4] > node2.pos[0] and node1.pos[4] < node2.pos[1]):
        return True
    if (node2.pos[4] > node1.pos[0] and node2.pos[4] < node1.pos[1]):
        return True

    return False


def if_correct_same_row(index_one, index_two, x):
    # print("index_one:min:{},mid:{},max:{}".format(x[index_one][2],x[index_one][5],x[index_one][3]))
    # print("index_two:min:{},mid:{},max:{}".format(x[index_two][2],x[index_two][5],x[index_two][3]))
    if (x[index_one][5] > x[index_two][2] and x[index_one][5] < x[index_two][3]):
        return True
    if (x[index_two][5] > x[index_one][2] and x[index_two][5] < x[index_one][3]):
        return True

    return False


# 将header分级
def classification(data, data_type, preds):
    row_relation = []

    x = data.x
    edges = data.edge_index
    row_relation = np.zeros((data.nodenum, data.nodenum), dtype=np.int)

    for i in range(len(preds)):
        s_node = edges[0][i]
        e_node = edges[1][i]
        # print("edges i:{}, s_node:{}, e_node:{}".format(i, s_node, e_node))
        if (if_correct_same_row(s_node, e_node, x)):
            # print("修改了")
            row_relation[s_node][e_node] = 1

    # 找出行极大团
    cliques = []
    maxn = 100

    # 分别是P集合，X集合，R集合
    some = np.zeros((maxn, maxn), dtype=np.int)
    none = np.zeros((maxn, maxn), dtype=np.int)
    r = np.zeros((maxn, maxn), dtype=np.int)

    for j in range(data.nodenum):
        some[1][j] = j + 1;

    adj_matrix = np.zeros((data.nodenum + 1, data.nodenum + 1), dtype=np.int)
    for p in range(data.nodenum):
        for q in range(data.nodenum):
            if (row_relation[p][q] == 1):
                adj_matrix[p + 1][q + 1] = 1

    dfs(1, 0, data.nodenum, 0, cliques, some, none, r, adj_matrix)
    for j in range(len(cliques)):
        for k in range(len(cliques[j])):
            cliques[j][k] -= 1

    avg_pos = []
    for i in range(len(cliques)):
        sum_pos = 0.0
        for j in range(len(cliques[i])):
            sum_pos += x[cliques[i][j]][5]
        avg_pos.append(deepcopy(sum_pos / len(cliques[i])))

    # 根据avg_pos对cliques排序
    high_to_low_quickSort(avg_pos, cliques, 0, len(cliques) - 1)

    # 对preds，edge_index与i的关系进行记录
    mark = [[0] * 50 for _ in range(50)]
    for i in range(len(preds)):
        s_node = edges[0][i]
        e_node = edges[1][i]
        mark[s_node][e_node] = i

    for i in range(len(preds)):
        s_node = edges[0][i]
        e_node = edges[1][i]
        preds[mark[e_node][s_node]] = preds[mark[s_node][e_node]]

    for i in range(len(cliques)):
        if (i == 0):
            # preds只能同行
            for p in range(len(cliques[i])):
                for q in range(len(cliques[i])):
                    if (p == q):
                        continue
                    preds[mark[cliques[i][p]][cliques[i][q]]] = 1
                    preds[mark[cliques[i][q]][cliques[i][p]]] = 1
            continue

        # preds位置同行则preds只能同行
        for j in range(len(cliques[i])):  # 对第i行的每一个节点

            for q in range(len(cliques[i])):
                if (q == j):
                    continue
                preds[mark[cliques[i][j]][cliques[i][q]]] = 1
                preds[mark[cliques[i][q]][cliques[i][j]]] = 1

            for k in range(i):  # 对第i行上方的每一行，该节点都只能与1个节点有1的关系，其余都是0
                tmp = []  # 存储有1关系的节点
                for m in range(len(cliques[k])):
                    if (preds[mark[cliques[i][j]][cliques[k][m]]] == 0):
                        continue
                    if (preds[mark[cliques[i][j]][cliques[k][m]]] == 1):
                        preds[mark[cliques[i][j]][cliques[k][m]]] == 0
                        preds[mark[cliques[k][m]][cliques[i][j]]] == 0
                    if (preds[mark[cliques[i][j]][cliques[k][m]]] == 2):
                        tmp.append(cliques[k][m])
                if (len(tmp) > 1):
                    final = tmp[0]
                    for n in range(len(tmp)):
                        if (n == 0):
                            continue
                        if (abs(x[cliques[i][j]][4] - x[tmp[n]][4]) < abs(x[cliques[i][j]][4] - x[final][4])):
                            preds[mark[cliques[i][j]][final]] = 0
                            preds[mark[final][cliques[i][j]]] = 0
                            final = tmp[n]
                        else:
                            preds[mark[cliques[i][j]][tmp[n]]] = 0
                            preds[mark[tmp[n]][cliques[i][j]]] = 0
    return preds