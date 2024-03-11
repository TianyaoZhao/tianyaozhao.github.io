## ssh

### 基本用法

**远程登录服务器：**

```bash
ssh user@hostname
```

- `user`：用户名
- `hostname`：IP地址或域名

**第一次登录时会提示：**

```bash
The authenticity of host '123.57.47.211 (123.57.47.211)' can't be established.
ECDSA key fingerprint is SHA256:iy237yysfCe013/l+kpDGfEG9xxHxm0dnxnAbJTPpG8.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入`yes`，然后`回车`即可

这样会将该服务器的信息记录在`~/.ssh/known_hosts`文件中

然后输入密码即可登录到远程服务器中

**退出当前服务器：**

```bash
logout
```

**默认登录端口号为22，如果想登录某一特定端口：**

```bash
ssh user@hostname -p 22
```

### 配置文件

**创建文件：**

```bash
~/.ssh/config
```

**在文件中输入：**

```shell
Host myserver1
    HostName IP地址或域名
    User 用户名

Host myserver2
    HostName IP地址或域名
    User 用户名
```

之后再使用服务器时，可以直接使用别名`myserver1、myserver2`

### 免密登录

**创建密钥：**

```bash
ssh-keygen
```

然后**一直回车**即可

**执行结束后，~/.ssh/目录下会多两个文件：**

- `id_rsa`：私钥
- `id_rsa.pub`：公钥

ㅤㅤ==之后想免密码登录哪个服务器，就将公钥传给哪个服务器即可==

ㅤㅤ例如，想免密登录`myserver`服务器。则将公钥中的内容，复制到`myserver`中的`~/.ssh/authorized_keys`文件里即可。

ㅤㅤ也可以使用如下命令**一键添加公钥**：

```
ssh-copy-id myserver
```



## scp

### 基本用法

![image-20231022143201946](./.assets/image-20231022143201946.png)

**`scp`传文件时，服务器后对应的路径是 `/`(最底层的路径)** 

**`/home/acs/xxxx`**

**`/root/xxx`**

**命令格式：**

将`source`路径下的文件复制到`destination`中

```bash
scp source destination
```

**一次复制多个文件：**

```bash
scp source1 source2 destination
```

**复制文件夹：**ㅤㅤ

将本地`家目录`中的`tmp`文件夹复制到`myserver`服务器中的`/home/acs/`目录下

```bash
scp -r ~/tmp myserver:/home/acs/
```

**mysever服务器没有root权限**

将`myserver`服务器中的`~/homework/`文件夹复制到本地的当前路径下

```bash
scp -r myserver:/home/acs/homework .
```

指定服务器的端口号：

```bash
scp -P 22 source1 source2 destination
```

注意： `scp`的`-r -P`等参数尽量加在`source`和`destination`之前。

**使用scp配置其他服务器的vim和tmux**

```bash
scp ~/.vimrc ~/.tmux.conf ~/.bashrc myserver:/home/acs/
```

