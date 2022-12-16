import os
import shutil


datelist=['20210108', '20210109', '20210110', '20210120', '20210125', '20210131', '20210210', '20210217', '20210220', '20210228', '20210304', '20210305', '20210315', '20210319', '20210330']


for datel in datelist:
    os.mkdir('outputs')
    newcommand=f'python T-A_API.py {datel} 20210101 20210331 0.15 0.30 0.55 region 21.838949 51.415695 -11.745695 23.145147'
    os.system(newcommand)
    os.rename("outputs", datel)




