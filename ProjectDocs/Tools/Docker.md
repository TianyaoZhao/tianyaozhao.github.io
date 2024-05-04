**docker**

https://docs.docker.com/engine/install/linux-postinstall/

## docker用户组

将当前用户添加到`docker`用户组

为了避免每次使用`docker`命令都需要加上`sudo`权限，可以将当前用户加入安装中自动创建的`docker`用户组

```bash
sudo usermod -aG docker $USER
```

## 镜像（images）

1. `docker pull ubuntu:20.04`：拉取一个镜像（名称:版本号）
2. `docker images`：列出本地所有镜像
3. `docker image rm ubuntu:20.04 或 docker rmi ubuntu:20.04`：删除镜像`ubuntu:20.04`
4. `docker [container] commit CONTAINER IMAGE_NAME:TAG`：创建某个`container`的镜像
5. `docker save -o ubuntu_20_04.tar ubuntu:20.04`：将镜像`ubuntu:20.04`导出到本地文件`ubuntu_20_04.tar`中 **还要加上可读权限 chmod + r ubuntu_20_04.tar**
6. `docker load -i ubuntu_20_04.tar`：将镜像`ubuntu:20.04`从本地文件`ubuntu_20_04.tar`中加载出来

## 容器（container）

1. `docker [container] create -it ubuntu:20.04`：利用镜像`ubuntu:20.04`创建一个容器。

2. `docker ps -a`：查看本地的所有容器状态

3. `docker [container] start CONTAINER`：启动容器

4. `docker [container] stop CONTAINER`：停止容器

5. `docker [container] restart CONTAINER`：重启容器

6. `docker [contaienr] run -itd ubuntu:20.04`：利用ubuntu20.04创建并启动一个容器

7. `docker [container] attach CONTAINER`：进入容器

8. **先按`Ctrl + p`，再按`Ctrl + q`可以挂起容器**

9. **`Ctrl + d`直接关掉容器**

10. `docker [container] exec CONTAINER COMMAND`：在容器中执行命令

11. `docker [container] rm CONTAINER`：删除容器

12. `docker container prune`：删除所有已停止的容器

13. `docker export -o xxx.tar CONTAINER`：将容器`CONTAINER`导出到本地文件`xxx.tar`中

14. `docker import xxx.tar image_name:tag`：将本地文件`xxx.tar`导入成镜像，并将镜像命名为`image_name:tag`

15. `docker export/import`与`docker save/load`的区别：

    `export/import`会丢弃历史记录和元数据信息，仅保存容器当时的快照状态

    `save/load`：会保存完整记录，体积更大

16. `docker top CONTAINER`：查看某个容器内的所有进程

17. `docker stats`：查看所有容器的统计信息，包括CPU、内存、存储、网络等信息

18. `docker cp xxx CONTAINER`:`xxx` 或 `docker cp CONTAINER:xxx xxx`：在本地和容器间复制文件

19. `docker rename CONTAINER1 CONTAINER2`：重命名容器

20. `docker update CONTAINER --memory 500MB`：修改容器限制





## 实战

进入AC Terminal，然后：

```bash
# 将镜像上传到自己租的云端服务器
scp /var/lib/acwing/docker/images/docker_lesson_1_0.tar server_name: 
# 登录自己的云端服务器
ssh server_name 
# 将镜像加载到本地
docker load -i docker_lesson_1_0.tar  
# 创建并运行docker_lesson:1.0镜像
docker run -p 20000:22 --name my_docker_server -itd docker_lesson:1.0  
# 进入创建的docker容器
docker attach my_docker_server 
# 设置root密码
passwd  
```

**去云平台控制台中修改安全组配置，放行端口20000。**

返回AC Terminal，即可通过ssh登录自己的docker容器：

`ssh root@xxx.xxx.xxx.xxx -p 20000`  # 将xxx.xxx.xxx.xxx替换成自己租的服务器的IP地址
然后，可以仿照上节课内容，创建工作账户acs。**这样root改为acs**

最后，可以参考ssh——ssh登录配置docker容器的别名和免密登录。

**创建用户**

`adduser acs`

**分配sudo权限**

`usermod -aG sudo acs`

**免密登录**

在本地服务器的 `~/.ssh/config`文件中添加docker的配置信息

```bash
Host myserver1
    HostName IP地址或域名
    User 用户名

Host myserver2_docker
    HostName IP地址(跟服务器的ip地址一样)
    User 用户名
    Port 20000
```

一键添加公钥

```bash
ssh-copy-id myserver2_docker
```

