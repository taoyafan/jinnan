# 环境要求

## 系统

ubuntu 16.04

## python 版本
python 3.5 或 python 3.6

## 需要的库

numpy, pandas, lightgbm, xgboost, sklearn



# 运行说明

生成 B 榜预测结果文件：
python3 run.py --test_type=B

生成 C 榜预测结果文件：
python3 run.py --test_type=C

# 其他说明

运行时间大概 6 分钟左右（不定）

要求一级目录下存放 data 文件夹, 内部有 A 榜数据： jinnan_round1_train_20181227.csv, B 榜数据： jinnan_round1_testB_20190121.csv, C 榜数据: jinnan_round1_test_20190121.csv.

生成结果在一级目录, B 榜名为 submit_B.csv, C 榜名为 submit_C.csv.

# 最后

提前祝您春节快乐～
如果有问题希望您联系我们

陶亚凡： 765370813@qq.com
Blue： cy_1995@qq.com 