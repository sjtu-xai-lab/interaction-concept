# Does a Neural Network Really Encode Symbolic Concepts?

PyTorch implementation of the ICML 2023 paper "Does a Neural Network Really Encode Symbolic Concepts?" [arxiv](https://arxiv.org/abs/2302.13080), a related [technical report](https://arxiv.org/abs/2304.13312).

## Requirements

PyTorch, NumPy, etc.

## Usage

~~~bash
cd ./src

# train model
python train_model_image_large.py --dataset=celeba_eyeglasses --arch=resnet34 \
  --batch-size=128 --lr=0.01 --logspace=1 --n-epoch=20 --seed=1 --gpu-id=0

# compute interaction
model_args="dataset=celeba_eyeglasses_model=resnet34_epoch=20_bs=128_lr=0.01_logspace=1_seed=0"
python eval_interaction_image.py --model-args=$model_args \
  --selected-dim=0-v0 --input=integrated_bg --baseline=zero --gpu-id=0 \
  --sparsify-loss=l1 --sparsify-qthres=0.04 --sparsify-qstd=vN-v0 \
  --sparsify-lr=1e-5 --sparsify-niter=10000

# repeat for all dataset/models
~~~

## Experiments

Please refer to `./experiments`

## Minimal example to compute interaction

~~~python
calculator = AndOrHarsanyi(
	model         = forward_func, 
    selected_dim  = selected_dim,
    x             = image, 
    baseline      = baseline, 
    y             = label,
    all_players   = all_players, 
    background    = background,
    mask_input_fn = mask_input_fn, 
    verbose       = 0
)
with torch.no_grad():
	calculator.attribute()

sparsifier = AndOrHarsanyiSparsifier(
    calculator = calculator, 
    **sparsify_kwargs
)
sparsifier.sparsify()
with torch.no_grad():
    I_and, I_or = sparsifier.get_interaction()
    I_and, I_or = reorganize_and_or_harsanyi(masks, I_and, I_or)
    sparsifier.save(save_folder=osp.join(save_folder, "interactions"))
~~~

## Citation

~~~late
@InProceedings{pmlr-v202-li23at,
  title={Does a Neural Network Really Encode Symbolic Concepts?},
  author={Li, Mingjie and Zhang, Quanshi},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={20452--20469},
  year={2023},
  volume={202},
  series={Proceedings of Machine Learning Research},
  month={23--29 Jul},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v202/li23at.html},
}


@article{li2023defining,
  title={Defining and Quantifying AND-OR Interactions for Faithful and Concise Explanation of DNNs},
  author={Li, Mingjie and Zhang, Quanshi},
  journal={arXiv preprint arXiv:2304.13312},
  year={2023}
}
~~~



