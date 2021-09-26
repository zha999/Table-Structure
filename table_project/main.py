
import os
import shutil
from table_ui.table_ui import Table_UI, batch_process
import argparse

sample_file = "/Users/hxr/PycharmProjects/GFTE/Test/img/page_1_tbl_1.png"
sample_dir = "/Users/hxr/PycharmProjects/GFTE/Test"

parser = argparse.ArgumentParser()
parser.add_argument('--ui', action='store_true', help='whether prase the table with UI')
parser.add_argument('--if_cuda', action='store_true', help='whether use cuda')
opt = parser.parse_args()


if __name__=="__main__":
    print("start table recognition.")
    if(opt is None):
        use_Ui = False # False  True
        if_cuda = False
    else:
        use_Ui = opt.ui
        if_cuda  = opt.if_cuda

    if use_Ui:
        ui = Table_UI(sample_file)
    else:
        batch_process(sample_dir, if_cuda)
    
    print("end.")




