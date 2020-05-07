---
categories: Data Structure
title: 剑指offer
date: 2020-04-26 18:12:29
tags: [Python, Data Structure]
---

# 数组
1. 数组占据一块连续的内存并且按照顺序存储，空间效率低
2. 哈希表：数组的下标设为哈希表的键值，数组中的数字为哈希表的值，“键值-值”的配对
3. 数组中重复的数字
	- set
	- list
4. 二维数组中的查找

# 字符串
1. 替换空格
	- `s.replace(' ','%20')`

# 链表
1. 链表是一种动态数据结构
2. 定义：
	```
	class ListNode:
	    def __init__(self, x):
	        self.val = x
	        self.next = None
	```
3. 从尾到头打印链表
	```
	class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
	    def printListFromTailToHead(self, listNode):
	        # write code here
	        l = []
	        head = listNode
	        while head:
	            l.insert(0, head.val)
	            head = head.next
	        return l
    ```

# 树
1. 二叉树
	- 前序遍历： 根-左-右
	- 中序遍历： 左-根-右
	- 后序遍历： 左-右-根
	- 定义：
	```
	class TreeNode:
	    def __init__(self, x):
	        self.val = x
	        self.left = None
	        self.right = None
	```
2. 重建二叉树
	- 前序遍历的第一个数字尾根节点，中序遍历中，根节点前的为左子树节点，后面为右子树
	- 可以用递归重建：
	```
	class Solution:
	    # 返回构造的TreeNode根节点
	    def reConstructBinaryTree(self, pre, tin):
	        # write code here
	        if len(pre)==0:
	            return None
	        if len(pre)==1:
	            return TreeNode(pre[0])
	        else:
	            root = TreeNode(pre[0])
	            r_idx = tin.index(pre[0])
	            root.left = self.reConstructBinaryTree(pre[1:r_idx+1],tin[0:r_idx])
	            root.right = self.reConstructBinaryTree(pre[r_idx+1:],tin[r_idx+1:])
	            return root
    ```
3. 二叉树的下一个节点

# 栈和队列
1. 栈：后进后出，最后被压入(push)栈的元素会第一个被弹出(pop)
2. 队列：先进先出，第一个进入队列的第一个出来。
3. 用两个栈实现队列：先将元素插入stack1,再将stack1中元素倒序插入stack2，从stack2中出栈：
	```
	class Solution:
	    def __init__(self):
	        self.stack1=[]
	        self.stack2=[]
	    def push(self, node):
	        # write code here
	        self.stack1.append(node)
	    def pop(self):
	        # return xx
	        if self.stack2==[]:
	            while self.stack1:
	                self.stack2.append(self.stack1.pop())
	        return self.stack2.pop()
    ```

# 递归和循环
1. 递归：在一个函数内部调用函数自身
2. 循环：重复运算
3. 