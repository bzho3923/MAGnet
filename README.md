# MAGnet

This repository is the official implementation of [Robust Graph Representation Learning for Local Corruption Recovery](https://download.arxiv.org/pdf/2202.04936v3.pdf).

![MAGnet](magnet_main.png)

## Requirements

To install requirements:

```
pip3 install -r requirements.txt
```

## Folder-generate_noisedata 
The folder generate_noisedata contains two types of noise, i.e. injecttion noise and noise by mettack. 
Also, the gae_run.py runs graph auto encoder to find the noisy item after the nosie generated on the feature matrix.

## Folder-denoise
Run the main_denoise.py will use the regularized optimization method to denoise the local corrupted featue matrix.

## Folder-class
Run the graph_class.py will test the performance on the denoised dataset.

## All Experiments
After sepecify the noise type and create noise on feature matrix, you can use the following command

```
sh run_all.sh
```
to run graph auto encoder, denosing and classification tasks.


## Citation 
If you consider our codes and datasets useful, please cite:
```
@inproceedings{zhou2022robust,
  title={Robust graph representation learning for local corruption recovery},
  author={Zhou, Bingxin and Jiang, Yuanhong and Wang, Yu Guang and Liang, Jingwei and Gao, Junbin and Pan, Shirui and Zhang, Xiaoqun},
  booktitle={The Web Conference},
  dio={https://doi.org/10.1145/3543507.3583399}
  year={2023}
}
```
