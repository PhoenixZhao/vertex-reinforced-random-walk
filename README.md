# Vertex Reinforced Random Walk for Network Embedding (reinforce2vec)

This repository provides a reference implementation of *reinforce2vec* as described in the paper in SDM 2020:<br>
> Vertex-reinforced Random Walk for Network Embedding.<br>
> Wenyi Xiao, Huan Zhao, Vincent W. Zheng, Yangqiu Song.<br>
> https://arxiv.org/abs/2002.04497 <Insert paper link>

We propose to use non-Markovian random walk, variants of vertex-reinforced random walk (VRRW), to fully use the history of a random walk path. To solve the getting stuck problem of VRRW, we introduce an exploitation-exploration mechanism to help the random walk jump out of the stuck set. *reinforce2vec* consists of two random walk models: VRRW and DRRW.

### Basic Usage

  
#### Node Classification
To run *drrw* on PPI, you can use the following command:<br/>
  ``python train.py --task unsupervised_node_classification --dataset ppi --model drrw --explore exploration``


#### Link Prediction
To run *drrw* on Facebook, you can use the following command:<br/>
 `` python link_prediction_my.py --model vrrw --input graphs/facebook.edgelist --explore exploration ``

#### Output
The probability of clicking an unseen document by the target user.

### Citing
If you find *reinforce2vec* useful for your research, please consider citing the following paper:

    @article{xiao2020vertex,
      title={Vertex-reinforced Random Walk for Network Embedding},
      author={Xiao, Wenyi and Zhao, Huan and Zheng, Vincent W and Song, Yangqiu},
      journal={arXiv preprint arXiv:2002.04497},
      year={2020}
    }


### Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <wxiaoae@cse.ust.hk>.
