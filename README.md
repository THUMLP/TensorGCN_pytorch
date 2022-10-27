# TensorGCN_pytorch
A pytorch version implementation of TensorGCN in paper:
Liu X, You X, Zhang X, et al. Tensor graph convolutional networks for text classification[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(05): 8409-8416.

For Tensorflow version, please click [here](https://github.com/THUMLP/TensorGCN).
# Requirement
+ python 3.6
+ pytorch 1.7.1
+ torch-geometric 1.7.2
# Reproduing Results
## Build three graphs

`python build_graph.py --gen_seq --gen_sem --gen_syn --dataset [dataset name]`

Here, [dataset name] could be 20ng, R8, R52, mr, ohsumed. You can also try your own dataset by processing your data similar to the data provided.
## Training and Evaluating

`python train.py --do_train --do_valid --do_test --dataset [dataset name]`
