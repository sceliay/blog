---
categories: Notes
title: Jupyter使用笔记
date: 2019-11-12 20:46:15
tags: [Jupyter]
---

使用jupter可以比较方便的写程序，并且在远程服务器上运行。
《[Jupyter Notebook介绍、安装及使用教程](https://zhuanlan.zhihu.com/p/33105153)》一文里比较全面的介绍了Jupter的使用方法。下面记录一下我需要的一些内容。

1. [远程连接服务器](https://blog.csdn.net/wangdan113269/article/details/88994792)
- 下载jupyter notebook。 比较方便的就是用[anaconda](https://www.anaconda.com/).

- 生成密钥：
```
$ ipython

from notebook.auth import passwd
passwd()
```
  输入密码后生成密钥，复制该密钥。

- 生成配置文件
```
$ jupyter notebook --generate-config
```

- 更改配置文件
```
$vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip='*'
c.NotebookApp.password = u'粘贴密钥'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888
```
端口号可根据自己服务器修改。保存后退出。

- 启动
```
$jupyter notebook --allow-root
```
在浏览器中打开
```
http://服务器IP:端口号
```

2. 修改主页面
```
$vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.notebook_dir = '主页面地址'
```