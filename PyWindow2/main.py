
import os
import shutil
from table_ui.table_ui import Table_UI, batch_process
import argparse

sample_file = "/Users/hxr/PycharmProjects/GFTE/Test/img/page_1_tbl_1.png"
sample_dir = "/Users/hxr/PycharmProjects/GFTE/Test"

parser = argparse.ArgumentParser()
parser.add_argument('--ui', action='store_true', help='whether prase the table with UI')




if __name__=="__main__":
    print("start table recognition.")

    use_Ui = False # False  True

    if use_Ui:
        ui = Table_UI(sample_file)
    else:
        batch_process(sample_dir)
    
    print("end.")




