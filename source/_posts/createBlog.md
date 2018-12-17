---
categories: 杂记
title: 用github+hexo搭建自己的博客
date: 2018-12-17 16:20:06
tags: [github,hexo,blog]
---

整个过程主要是参考了[我是如何利用Github Pages搭建起我的博客，细数一路的坑](https://www.cnblogs.com/jackyroc/p/7681938.html)。但是在搭建的过程中，还是遇到了很多问题，所以写个笔记记录一下。

### 一. 创建Github Pages
这一步可以参考[原文](https://www.cnblogs.com/jackyroc/p/7681938.html)

### 二. 安装Hexo
1. 首先安装[Node.js](https://nodejs.org/en/download/)和[Git](https://git-scm.com/download/)
2. 使用Git Bash安装Hexo:
` npm install hexo-cli -g `
` hexo init blog //初始化网站，文件夹名为“blog” `
` npm install `   
` hexo g //hexo generate, 生成静态页面 `
` hexo s //hexo server,启动本地服务器，这一步之后就可以通过http://localhost:4000查看 `
3. 创建文章
` hexo new 'post' //post为文章名字`
4. 创建页面
` hexo new 'page' //page为页面名字`
5. 添加主题
以'yilia'为例，先下载主题，存放到themes文件夹下:
` hexo clean`
` git clone https://github.com/litten/hexo-theme-yilia.git themes/yilia `
找到目录下的_config.yml 文件,打开找到 theme：属性并设置为主题的名字（yilia)。
更多主题可以在[https://hexo.io/themes/](https://hexo.io/themes/)上面找
6. 部署到github上
原文说的` hexo d `，我使用后并没有什么用。然后很暴力的把`hexo g`生成的public文件夹里面的所有内容，放到'*.github.io'的目录下，然后更新到github上面，就可以了。虽然更新有点麻烦，不过至少可以用了。如果有更好的办法，可以私信给我嘛~
