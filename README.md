# 文本续写工具

## 功能
给定一段文本，模型会给出续写内容，续写长度为10个文字，你可以接纳也可以不接纳。直到你输入exit退出推荐

## 方法
langchain+通译大模型
使用通译千问的agent大模型来完成工作

## 评估
测试集：一些公开的新闻数据，https://huggingface.co/datasets/RealTimeData/bbc_news_alltime，从中每个月份抽取15条文本组成训练集，提供每个新闻content内容的前半段，来评估续写的内容和后半段内容是否匹配
