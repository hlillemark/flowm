# Flow Equivariant World Models (FloWM)
Welcome to the codebase for the paper [Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments](https://flowequivariantworldmodels.github.io/)! This repository contains the code and commands for training and running inference on FloWM models in the 2D (MNIST World) and 3D (Dynamic Block World) environments. It also contains the code for running the Diffusion Forcing Transformer (DFoT) and Diffusion State Space Model (DFoT-SSM) baselines. Please follow the steps below to run experiments. 

Contents:
- [Environment Setup](#environment)
- [Dataset Download](#dataset)
- [Model Checkpoint Download](#model-checkpoints)
- [Inference commands (replicating results)](#inference-commands)
- [Training commands](#training-commands)
- [Code Walkthrough](#code-walkthrough)
- [Citation](#citation)

# Setup

## Environment

This command will create a conda environment named `flowm`. If you are using cuda version 13, then please also attach the flag `--cuda13`. The other default is `--cuda12.4`. If the flash attention install breaks, see [here for more help](https://github.com/Dao-AILab/flash-attention/issues/1708#issuecomment-2995556229):

```
bash ./setup_flowm_env.sh
```

If you would like to run the `dfot-ssm` baseline, then please run the following command instead: This script is more fragile and you may need to look into the issues [for causal_conv1d here](https://github.com/state-spaces/mamba/issues/55#issuecomment-1858638484) and [for mamba ssm here](https://github.com/state-spaces/mamba/issues/719).
```
bash ./setup_flowm_env.sh --download-ssm
conda activate flowm
```

To track runs on wandb, make a copy of `configurations/my_secrets_template.yaml`, rename it `my_secrets.yaml`, and then fill in your wandb api key there. You will also need to fill in `entity` and `project` in `config.yaml`.

## Dataset

To download the datasets used in the paper, please use the script `./download_datasets.sh`. This script specifies the dataset, split, and configuration. To download all splits and all data, use the following command:

```bash
bash ./download_datasets.sh --download-workers 16 --extract-workers 16
```

See the file for more options. To download just the `blockworld tex validation` split, run the following:

```bash
bash ./download_datasets.sh --dataset blockworld --configs tex --splits validation --download-workers 16 --extract-workers 16
```

After downloading, the datasets will appear in the `./data/` folder. When running the code, metadata will be automatically generated. If you only downloaded a subset of the data, then please delete the corresponding metadata for the split you are now downloading (which will be generated as empty). For instance, if you only downloaded the `tex blockworld validation` split and now want to download the `tex blockworld training` split, delete `./data/blockworld/metadata/tex_training.pt` before running the training code.

Each configuration has `training` and `validation` splits. The `blockworld` dataset has configs `dynamic` (main results in paper), `static`, and `tex`. The `mnist_world` dataset has configs `dynamic_po` (main results in paper), `static_po`, `dynamic_fo`, and `dynamic_fo_no_sm`, where `po` means partially observed, `fo` means fully observed, and `no_sm` means no self motion. Change the configurations to the script accordingly to download the splits of need. Only the validation splits are necessary to replicate the paper's results.

For the code to generate the 3D Dynamic Blockworld dataset, please see this repo: [Link](https://github.com/hlillemark/Miniworld).

## Model Checkpoints
If intending to run from scratch, you may skip this part (except the VAE if training DFoT or DFoT-SSM from scratch). We provide checkpoints for the primary result splits.

The following script will download all checkpoints for all models for all datasets (FloWM, DFoT, DFoT-SSM, VAE), requiring ~15GB of space: 

```bash
bash ./download_ckpts.sh
```

### Checkpoint map
To keep track of downloaded checkpoints, we provide a default checkpoint map at `configurations/ckpt_map/default.yaml`. This creates a pointer to downloaded checkpoints when the code actually runs. Downloading from the above script should work automatically, but look into this file if you find any errors.

# Inference Commands 

Please see [the wiki](https://github.com/hlillemark/flowm/wiki/Inference-and-Reproducing-Results) for full commands! For example, you can run inference on the Textured Blockworld Validation set for 280 frames using FloWM with the following command: 

```bash
python -m main shortcode=exp/blockworld/flowm/infer/metrics_140/tex_70ctx +name=infer_blockworld_flowm_tex_70ctx_140 algorithm=flowm_video dataset=blockworld ckpt_map=default
```



# Training Commands

Please see [the wiki](https://github.com/hlillemark/flowm/wiki/Training) for full commands! For example, you can train FloWM on the Textured Blockworld Training set with the following command:

```bash
python -m main shortcode=exp/blockworld/flowm/train/tex +name=blockworld_flowm_10m_tex_50c90p algorithm=flowm_video dataset=blockworld ckpt_map=default
```


# Code Walkthrough

Please see our [wiki](https://github.com/hlillemark/flowm/wiki) for more information on the code itself. This wiki includes information on the configurations, FloWM and baseline algorithm implementations, VAE information, adding a new dataset, and more.


# Citation

If you find our work interesting or helpful, please cite it using the following BibTex:

```bibtex
@misc{lillemark2026flowequivariantworldmodels,
    title={Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments}, 
    author={Hansen Jin Lillemark and Benhao Huang and Fangneng Zhan and Yilun Du and Thomas Anderson Keller},
    year={2026},
    eprint={2601.01075},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2601.01075}, 
  }
```

# Acknowledgement

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). 
