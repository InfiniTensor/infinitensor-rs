﻿# 设计思路

## 图变换规则的存储问题

*图变换*指的是优化过程中，用一个图替换另一个图的过程。优化的根本原理就是图变换。具体来说，就是在表示完整 AI 程序的图中找到要替换的子图，然后将作为目标的另一个图连接到这个子图整体输入输出的张量上。

图变换中要替换的子图和目标图的关系称为*规则*。

通常来说，图的结构包含两部分，算子的连接关系和每个算子持有的张量信息。然而在规则中子图的输入输出都是已知的，张量的信息都可以推断出来，因此不需要存储。只要支持一种没有填写张量信息的图就能用数据而不是程序存储规则了。张量信息可以等替换进去再填写，填写的逻辑只需要一套。
