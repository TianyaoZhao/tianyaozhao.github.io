- 显示工作空间

   `conda info --env`   ` conda info -e ` 

- 创建工作空间

   `conda create --name <自定义的名字> python=3.9`

   `conda create -n <自定义的名字> python=3.9`

- 删除工作空间

- `conda remove --name <自定义的名字> --all`

   `conda remove -n <自定义的名字> --all`

- 回滚工作空间

   `conda list --revisions`
   `conda install --rev 0`

- 安装包

   `pip install <>`

   `conda install <>`

- 删除包

   `pip uninstall <>`

   `conda uninstall <>`

- 显示安装的包

   `pip list`

   `conda list`

- anaconda换清华源

   [Anaconda 换国内源_conda 换源_scl52tg的博客-CSDN博客](https://blog.csdn.net/scl52tg/article/details/120959893)

   `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   conda config --set show_channel_urls yes`

- 恢复默认源

  `conda config --remove-key channels`

- 查看源

  `conda config --show`

- pip换源

  [python pip 换国内源的办法（永久和临时两种办法）_pip换源_skyyzq的博客-CSDN博客](https://blog.csdn.net/skyyzq/article/details/113417832)

   	[Python 修改 pip 源为国内源 - 点点米饭 - 博客园 (cnblogs.com)](https://www.cnblogs.com/137point5/p/15000954.html)

​		`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

     1 error detected in the compilation of "bevdepth/ops/voxel_pooling_train/src/voxel_pooling_train_forward_cudxBuilt on Mon_May__3_19:15:13_PDT_2021
    error: command '/usr/local/cuda/bin/nvcc' failed with exit code 1    
