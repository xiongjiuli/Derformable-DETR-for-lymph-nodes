# Deformable DETR for lymph nodes 

Deformable DETR  is by [Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ),  [Weijie Su](https://www.weijiesu.com/),  [Lewei Lu](https://www.linkedin.com/in/lewei-lu-94015977/), [Bin Li](http://staff.ustc.edu.cn/~binli/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Jifeng Dai](https://jifengdai.org/).

This repository is an official implementation of the paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

We use it for detecting lymph nodes as an transformer-based method. We slice our CT images, perform detection on each slice, and then merge the slices into a 3D volume. However, there are too many false positives in single CT scans, although the method performs well on the slices.
