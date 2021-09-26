# 插入data
# 根据最底层header的x坐标对应列数
# 根据最右侧attributer的y坐标对应行数
# 为了区分，data部分用t_data表示
# t_data是t_data部分的node类列表
# max_header_id是最底层header的node id列表
# max_attr_id是最右侧attributer的node id列表

import copy


# 快速排序只处理列表版
def n_partition(arr, low, high, index):
    i = (low - 1)  # 最小元素索引
    pivot = arr[high]

    for j in range(low, high):

        # 当前元素小于或等于 pivot
        if arr[j].pos[index] <= pivot.pos[index]:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    return (i + 1)


# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引

# 快速排序函数
def n_quickSort(arr, low, high, index):
    if low < high:
        pi = n_partition(arr, low, high, index)

        n_quickSort(arr, low, pi - 1, index)
        n_quickSort(arr, pi + 1, high, index)


def if_same_col(node1, node2):
    if (node1.pos[4] > node2.pos[0] and node1.pos[4] < node2.pos[1]):
        return True
    if (node2.pos[4] > node1.pos[0] and node2.pos[4] < node1.pos[1]):
        return True

    return False


def if_same_row(node1, node2):
    if (node1.pos[5] > node2.pos[2] and node1.pos[5] < node2.pos[3]):
        return True
    if (node2.pos[5] > node1.pos[2] and node2.pos[5] < node1.pos[3]):
        return True

    return False



def insert_data(t_data, header, attributer, max_header, max_attr):
    # print("t_data:{}".format(t_data))
    # print("t_data:{}".format(header))

    col_wait_left = []  # 待处理在最前面的节点
    col_wait_mid = []  # 待处理在中间的节点
    col_wait_mid_hindex = []  # 记录夹在中间的节点的左侧max header index
    col_wait_right = []  # 待处理在后面的节点
    col_proceed_node = []  # 在中心点对应步骤中就处理完毕的节点

    # 首先确定列数
    # 对data部分按照x中心点坐标排序
    n_quickSort(t_data, 0, len(t_data) - 1, 4)
    n_quickSort(max_header, 0, len(max_header) - 1, 4)

    # 根据中心点坐标对应header

    '''
    max_header_id是最底层header的id列表【3，8，9，20】
    想要快速获取这个节点的一切信息，最好传进来的时候就是node型的
    '''

    t_data_number = 0  # 记录待处理列表共有几列
    t_col_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

    for i in range(len(t_data) - 1):
        # print("t_data[i].tex:{},t_data[i].pos:{}".format(t_data[i].tex, t_data[i].pos))
        # print("t_data[i+1].tex:{},t_data[i+1].pos:{}".format(t_data[i+1].tex,t_data[i+1].pos))
        '''
        print("t_data[i]:{}".format(t_data[i].tex))
        print("t_data[i]中心点:{},左侧:{},右侧:{}".format(t_data[i].pos[4], t_data[i].pos[0], t_data[i].pos[1]))
        print("t_data[i+1]:{}".format(t_data[i+1].tex))
        print("t_data[i]中心点:{},左侧:{},右侧:{}".format(t_data[i+1].pos[4], t_data[i+1].pos[0], t_data[i+1].pos[1]))
        '''
        if (if_same_col(t_data[i], t_data[i + 1])):
            continue
        else:
            # print("换列了！")
            t_data_number += 1
            t_col_divide_index.append(i)

    t_col_divide_index.append(len(t_data) - 1)
    # col_wait_numer = len(col_divide_index)
    t_data_number += 1

    for i in range(t_data_number):

        # 首先查找是否有能够对应的header
        flag_corr = False
        for j in range(len(max_header)):
            if (if_same_col(t_data[t_col_divide_index[i]], max_header[j])):
                for k in range(t_col_divide_index[i] + 1):
                    if (i != 0 and k <= t_col_divide_index[i - 1]):
                        continue
                    t_data[k].start_col = max_header[j].start_col
                    t_data[k].end_col = max_header[j].end_col
                    # print("k:{}".format(k))
                flag_corr = True
                break

        # 如果没有对应的header查找属于三种情况的哪一种
        if (not flag_corr):

            xmin = 10000
            xmax = 0
            for k in max_header:
                if (k.pos[0] < xmin):
                    xmin = k.pos[0]
                if (k.pos[1] > xmax):
                    xmax = k.pos[1]

            # 是否在最左侧
            flag_left = False
            if (t_data[t_col_divide_index[i]].pos[4] < xmin):
                flag_left = True
                # 将这列都投入到待处理列表中
                for k in range(t_col_divide_index[i] + 1):
                    if (i != 0 and k <= t_col_divide_index[i - 1]):
                        continue
                    t_data[k].start_col = -1
                    t_data[k].end_col = -1
                    col_wait_left.append(t_data[k])

            # 是否在最右侧
            if (not flag_left):
                flag_right = False
                if (t_data[t_col_divide_index[i]].pos[4] > xmax):
                    flag_right = True
                # 将这列都投入到待处理列表中
                for k in range(t_col_divide_index[i] + 1):
                    if (i != 0 and k <= t_col_divide_index[i - 1]):
                        continue
                    t_data[k].start_col = -1
                    t_data[k].end_col = -1
                    col_wait_right.append(t_data[k])

            # 夹在中间
            if ((not flag_left) and (not flag_right)):
                flag_mid = False
                for k in range(len(max_header) - 1):
                    if (t_data[t_col_divide_index[i]].pos[4] > max_header[k].pos[1] and
                            t_data[t_col_divide_index[i]].pos[4] < max_header[k + 1].pos[0]):
                        flag_mid = True
                        # 将这列都投入到待处理列表中
                        for q in range(t_col_divide_index[i] + 1):
                            if (i != 0 and q <= t_col_divide_index[i - 1]):
                                continue
                            t_data[q].start_col = -1
                            t_data[q].end_col = -1
                            col_wait_mid.append(t_data[q])
                        col_wait_mid_hindex.append(k)

                    break

    '''
    ***************处理col_wait_left***************
    '''
    if (len(col_wait_left) != 0):
        col_wait_number = 0  # 记录待处理列表共有几列
        col_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(col_wait_left) - 1):
            if (if_same_col(col_wait_left[i], col_wait_left[i + 1])):
                continue
            else:
                col_wait_number += 1
                col_divide_index.append(i)

        col_divide_index.append(len(col_wait_left) - 1)
        # col_wait_numer = len(col_divide_index)
        col_wait_number += 1

        # 一列一列处理

        for i in range(col_wait_number):
            candidate_header = []  # 最底层header的上层节点可能覆盖新出现的data列

            for k in range(len(header)):

                if (header[k].start_col == 0):
                    candidate_header.append(header[k])
                if (header[k].start_col != 0):
                    header[k].start_col += 1
                    header[k].end_col += 1

            for k in range(len(t_data)):
                if (t_data[k].start_col != -1):
                    t_data[k].start_col += 1
                    t_data[k].end_col += 1

            for k in range(len(candidate_header)):
                if (if_same_col(col_wait_left[col_divide_index[i]], candidate_header[k])):
                    candidate_header[k].start_col = 0
                    candidate_header[k].end_col += 1

            for k in range(col_divide_index[i] + 1):
                if (i != 0 and k <= col_divide_index[i - 1]):
                    continue
                col_wait_left[k].start_col = i
                col_wait_left[k].end_col = i

    '''
    ***************处理col_wait_mid***************
    '''
    if (len(col_wait_mid) != 0):
        col_wait_number = 0  # 记录待处理列表共有几列
        col_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(col_wait_mid) - 1):
            if (if_same_col(col_wait_mid[i], col_wait_mid[i + 1])):
                continue
            else:
                # print("i:{}".format(i))
                col_wait_number += 1
                col_divide_index.append(i)

        col_divide_index.append(len(col_wait_mid) - 1)
        # col_wait_numer = len(col_divide_index)
        col_wait_number += 1

        # 一列一列处理

        for i in range(col_wait_number):

            # 6.24 last_index = col_divide_index[i] #待处理节点 第i列 最后一个节点在待处理列表中的index
            end_col = max_header[col_wait_mid_hindex[i]].end_col

            for k in range(len(header)):
                if (header[k].start_col <= end_col):
                    pass
                else:
                    header[k].start_col += 1
                if (header[k].end_col <= end_col):
                    pass
                else:
                    header[k].end_col += 1

            for k in range(len(t_data)):
                if (t_data[k].start_col != -1):
                    if (t_data[k].start_col <= end_col):
                        pass
                    else:
                        t_data[k].start_col += 1
                    if (t_data[k].end_col <= end_col):
                        pass
                    else:
                        t_data[k].end_col += 1

            for k in range(col_divide_index[i] + 1):
                # print("len(col_wait_mid):{}".format(len(col_wait_mid)))
                # print("k:{}".format(k))
                if (i != 0 and k <= col_divide_index[i - 1]):
                    continue
                col_wait_mid[k].start_col = end_col + 1
                col_wait_mid[k].end_col = end_col + 1
                # end_col += 1

    '''
    ***************处理col_wait_right***************
    '''
    if (len(col_wait_right) != 0):
        col_wait_number = 0  # 记录待处理列表共有几列
        col_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(col_wait_right) - 1):
            if (if_same_col(col_wait_right[i], col_wait_right[i + 1])):
                continue
            else:
                col_wait_number += 1
                col_divide_index.append(i)

        col_divide_index.append(len(col_wait_right) - 1)
        # col_wait_numer = len(col_divide_index)
        col_wait_number += 1

        # 一列一列处理

        col_max = 0
        for k in range(len(header)):
            if (header[k].end_col > col_max):
                col_max = header[k].end_col

        for i in range(col_wait_number):

            candidate_header = []  # 最底层header的上层节点可能覆盖新出现的data列

            for k in range(len(header)):
                if (header[k].end_col > col_max):
                    col_max = end_col

            for k in range(len(header)):
                if (header[k].start_col == col_max):
                    candidate_header.append(header[k])

            for k in range(len(candidate_header)):
                if (if_same_col(col_wait_right[col_divide_index[i]], candidate_header[k])):
                    candidate_header[k].end_col += 1

            for k in range(col_divide_index[i] + 1):
                if (i != 0 and k <= col_divide_index[i - 1]):
                    continue
                col_wait_right[k].start_col = col_max + 1
                col_wait_right[k].end_col = col_max + 1

            col_max += 1

    '''
    ***************确定行数********************
    '''

    row_wait_up = []  # 待处理在最前面的节点
    row_wait_mid = []  # 待处理在中间的节点
    row_wait_mid_aindex = []  # 记录夹在中间的节点的上侧 max attributer index
    row_wait_down = []  # 待处理在后面的节点

    # 对data部分按照y中心点坐标排序
    n_quickSort(t_data, 0, len(t_data) - 1, 5)  # 注意这里y是从小到大排列的，即行数是从大到小的
    n_quickSort(max_attr, 0, len(max_attr) - 1, 5)

    # 根据中心点坐标对应attributer

    t_data_row_number = 0  # 记录待处理列表共有几行
    t_row_divide_index = []  # 记录当前行的最后一个节点在待处理列表中的index

    for i in range(len(t_data) - 1):
        if (if_same_row(t_data[i], t_data[i + 1])):
            continue
        else:
            t_data_row_number += 1
            t_row_divide_index.append(i)

    t_row_divide_index.append(len(t_data) - 1)
    t_data_row_number += 1

    for i in range(t_data_row_number):

        # 首先查找是否有能够对应的attributer
        flag_corr_row = False
        for j in range(len(max_attr)):
            if (if_same_row(t_data[t_row_divide_index[i]], max_attr[j])):
                # print("对应的attr:{}对应的data:{}".format(max_attr[j].tex, t_data[t_row_divide_index[i]].tex))
                for k in range(t_row_divide_index[i] + 1):
                    if (i != 0 and k <= t_row_divide_index[i - 1]):
                        continue
                    t_data[k].start_row = max_attr[j].start_row
                    t_data[k].end_row = max_attr[j].end_row
                    # print("s{}e{}".format(max_attr[j].start_row, max_attr[j].end_row))
                flag_corr_row = True
                break

        # 如果没有对应的attributer查找属于三种情况的哪一种
        if (not flag_corr_row):

            ymin = 1000000
            ymax = 0
            for k in max_attr:
                if (k.pos[2] < ymin):
                    ymin = k.pos[2]
                if (k.pos[3] > ymax):
                    ymax = k.pos[3]

            # 是否在最上方
            flag_up = False
            if (t_data[t_row_divide_index[i]].pos[5] > ymax):
                flag_up = True
                # 将这列都投入到待处理列表中
                for k in range(t_row_divide_index[i] + 1):
                    if (i != 0 and k <= t_row_divide_index[i - 1]):
                        continue
                    t_data[k].start_row = -1
                    t_data[k].end_row = -1
                    row_wait_up.append(t_data[k])
                # print("存在row_wait_up")

            # 是否在最下侧
            if (not flag_up):
                flag_down = False
                if (t_data[t_row_divide_index[i]].pos[5] < ymin):
                    flag_down = True
                    # 将这行都投入到待处理列表中
                    for k in range(t_row_divide_index[i] + 1):
                        if (i != 0 and k <= t_row_divide_index[i - 1]):
                            continue
                        t_data[k].start_row = -1
                        t_data[k].end_row = -1
                        row_wait_down.append(t_data[k])

            # 夹在中间
            if ((not flag_up) and (not flag_down)):
                flag_row_mid = False
                for k in range(len(max_attr) - 1):
                    if (t_data[t_row_divide_index[i]].pos[5] > max_attr[k].pos[2] and t_data[t_row_divide_index[i]].pos[
                        5] < max_attr[k + 1].pos[3]):
                        flag_row_mid = True
                        # 将这列都投入到待处理列表中
                        for q in range(t_row_divide_index[i] + 1):
                            if (i != 0 and q <= t_row_divide_index[i - 1]):
                                continue
                            t_data[q].start_row = -1
                            t_data[q].end_row = -1
                            row_wait_mid.append(t_data[q])
                        row_wait_mid_aindex.append(k + 1)

                    break

    '''
    ***************处理row_wait_up***************
    '''

    if (len(row_wait_up) != 0):
        # print("处理row_wait_up")

        row_wait_number = 0  # 记录待处理列表共有几列
        row_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(row_wait_up) - 1):
            if (if_same_row(row_wait_up[i], row_wait_up[i + 1])):
                continue
            else:
                row_wait_number += 1
                row_divide_index.append(i)

        row_divide_index.append(len(row_wait_up) - 1)
        # row_wait_numer = len(row_divide_index)
        row_wait_number += 1

        # 一行一行处理

        for i in range(row_wait_number):

            i = row_wait_number - i - 1

            candidate_attr = []  # 最右侧attributer的上层节点可能覆盖新出现的data列

            for k in range(len(attributer)):

                if (attributer[k].start_row == 0):
                    candidate_attr.append(attributer[k])
                if (attributer[k].start_row != 0):
                    attributer[k].start_row += 1
                    attributer[k].end_row += 1

            for k in range(len(t_data)):
                if (t_data[k].start_row != -1):
                    t_data[k].start_row += 1
                    t_data[k].end_row += 1

            for k in range(len(candidate_attr)):
                if (if_same_row(row_wait_up[row_divide_index[i]], candidate_attr[k])):
                    candidate_attr[k].start_row = 0
                    candidate_attr[k].end_row += 1

            for k in range(row_divide_index[i] + 1):
                if (i != 0 and k <= row_divide_index[i - 1]):
                    continue
                row_wait_up[k].start_row = i
                row_wait_up[k].end_row = i

    '''
    ***************处理row_wait_mid***************
    '''
    if (len(row_wait_mid) != 0):
        row_wait_number = 0  # 记录待处理列表共有几列
        row_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(row_wait_mid) - 1):
            if (if_same_row(row_wait_mid[i], row_wait_mid[i + 1])):
                continue
            else:
                # print("i:{}".format(i))
                row_wait_number += 1
                row_divide_index.append(i)

        row_divide_index.append(len(row_wait_mid) - 1)
        # row_wait_numer = len(row_divide_index)
        row_wait_number += 1

        # print("row_divide_index:{}".format(row_divide_index))
        # print("row_wait_number:{}".format(row_wait_number))
        # for i in row_wait_mid:
        # print("row_wait_mid:{}".format(i.tex))

        # 一行一行处理

        for i in range(row_wait_number):

            i = row_wait_number - i - 1

            # last_index = row_divide_index[i] #待处理节点 第i行 最后一个节点在待处理列表中的index
            # end_row = max_attr[row_wait_mid_aindex[last_index]].end_row #该行待处理节点的上层attr
            end_row = max_attr[row_wait_mid_aindex[i]].end_row

            for k in range(len(attributer)):
                if (attributer[k].start_row <= end_row):
                    pass
                else:
                    attributer[k].start_row += 1
                if (attributer[k].end_row <= end_row):
                    pass
                else:
                    attributer[k].end_row += 1

            for k in range(len(t_data)):
                if (t_data[k].start_row != -1):
                    if (t_data[k].start_row <= end_row):
                        pass
                    else:
                        t_data[k].start_row += 1
                    if (t_data[k].end_row <= end_row):
                        pass
                    else:
                        t_data[k].end_row += 1

            for k in range(row_divide_index[i] + 1):
                if (i != 0 and k <= row_divide_index[i - 1]):
                    continue
                row_wait_mid[k].start_row = end_row + 1
                row_wait_mid[k].end_row = end_row + 1
                # end_row += 1

    '''
    ***************处理row_wait_down***************
    '''
    if (len(row_wait_down) != 0):
        # print("处理row_wait_down")
        row_wait_number = 0  # 记录待处理列表共有几列
        row_divide_index = []  # 记录当前列的最后一个节点在待处理列表中的index

        for i in range(len(row_wait_down) - 1):
            if (if_same_row(row_wait_down[i], row_wait_down[i + 1])):
                continue
            else:
                row_wait_number += 1
                row_divide_index.append(i)

        row_divide_index.append(len(row_wait_down) - 1)
        # col_wait_numer = len(col_divide_index)
        row_wait_number += 1

        '''
        print("总共有{}行".format(row_wait_number))
        print("row_divide_index的长度为:{}".format(len(row_divide_index)))
        print("len(row_wait_down:{}".format(len(row_wait_down)))
        '''

        row_max = 0
        for k in range(len(attributer)):
            if (attributer[k].end_row > row_max):
                row_max = attributer[k].end_row
        # 一行一行处理
        for i in range(row_wait_number):

            i = row_wait_number - i - 1

            candidate_attr = []  # 最右侧的attr的上层节点可能覆盖新出现的data列

            for k in range(len(attributer)):
                if (attributer[k].end_row > row_max):
                    row_max = attributer[k].end_row


            for k in range(len(attributer)):
                if (attributer[k].end_row > row_max):
                    candidate_attr.append(attributer[k])

            for k in range(len(candidate_attr)):

                if (if_same_row(row_wait_down[row_divide_index[i]], candidate_attr[k])):
                    candidate_attr[k].end_row += 1

            # 由于y的排序是从小到大排序的，行数是从大到小
            for k in range(row_divide_index[i] + 1):
                if (i != 0 and k <= row_divide_index[i - 1]):
                    continue
                row_wait_down[k].start_row = row_max + 1
                row_wait_down[k].end_row = row_max + 1

            row_max += 1

    return header, attributer, t_data