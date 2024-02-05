## Deeper Understanding of Black-box Predictions via Generalized Influence Functions

This repository is the official implementation of [Deeper Understanding of Black-box Predictions via Generalized Influence Functions](https://arxiv.org/abs/2312.05586). 

## Requirements

This repository is conda-based. Please install Conda first, then precede the installation.
To install requirements:

```setup
conda env create -f conda_env.yaml
conda activate GIF
cd GIF && source python_path.sh
```

## Training

You can train the model by using the train.py. Please change the preset in the config.py.

```train
python train.py
```

## Results

All figures and tables can be reproduced by the notebook in the `scripts`


## Acknowledgements

I referred to the following repos:

* Models: [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* Lanczos method:[noahgolmant/pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings)

## Reference

```
@misc{lyu2023deeper,
      title={Deeper Understanding of Black-box Predictions via Generalized Influence Functions}, 
      author={Hyeonsu Lyu and Jonggyu Jang and Sehyun Ryu and Hyun Jong Yang},
      year={2023},
      eprint={2312.05586},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
