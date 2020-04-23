---
categories: Notes
title: 服务器环境命令
date: 2020-01-06 16:17:43
tags: [Anaconda,tmux]
---

记录一些服务器上常用命令。

# Anaconda
- 添加环境
`export PATH="/home/ry/anaconda3/bin:$PATH"`
- Python 3 与 Python 2 环境转换
1. 创建环境
`conda create -n python27 python=2.7 anaconda`
2. 进入环境
`source activate env_name`
3. 离开环境
`source deactivate`
4. 列出环境
`conda env list`
5. 删除环境
`conda env remove -n env_name`
6. 导出环境
`conda env export > environment.yaml`
7. 加载环境
`conda env create -f environment.yaml`

# Tmux
0. 安装
`sudo apt install tmux`
1. 创建会话
`tmux new -s name`
2. 分离会话，退出当前tmux窗口，使之在后台运行
`tmux detach` 或 `Ctrl+b d`
3. 查看所有会话
`tmux ls` 或 `tmux list-session`
4. 接入会话
`tmux a -t name` 或 `tmux attach -t name`
5. 杀死会话
`tmux kill-session -t name`
6. 切换会话
`tmux switch -t name`
7. 重命名会话
`tmux rename-session -t name`

# VNC
1. 创建会话
`vncserver`
2. 设置密码
`vncpasswd`
3. 登录
`ip:server id`