# AdaptiveDiffuseMotion

**[Project Page](https://symbolzzz.github.io/AdaptiveDiffuseMotion/)** | **[Paper](https://ieeexplore.ieee.org/document/11463428)** | **[Code](https://github.com/Symbolzzz/AdaptiveDiffuseMotion)**

Adaptive Multi-Task Diffusion Model for Speech-Driven Holistic Motion Generation

## Getting Started

First, update the `prefix` section in `environment.yaml` to match your directory path, then set up the environment by running:

```bash
conda env create -f environment.yaml
conda activate ADM
```

## Inference

1.	Download the WavLM Large model from [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) .
2.	Download the FastText model from [crawl-300d-2M.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip).

Then, download our pre-trained model from Google Drive, replace the paths in `./sys.sh` with your model paths, and run:

```bash
./sys.sh
```

## Training Your Model

1.	Download the [BEAT](https://github.com/PantoMatrix/PantoMatrix) dataset and unzip it to `./data/source/`. The dataset includes data from 30 speakers, and you can select a subset of them for training. The source folder should be organized as follows:

```bash
├── data
│   └── source
│       ├── 1
│       │   └── data ...
│       ├── 2
│       └── ...
```

2.	Update the file paths in `process_data.sh` to match your directory structure, then follow the steps `step1`, `step3`, and `step4` to process the data and create `.h5` files.

```bash
./process_data.sh
```

After `step4` is done, run:

```bash
cd process
python calculate_gesture_statistics.py --dataset BEAT --version "v0"
```

3.	In `./configs/AdaptiveDiffuseMotion.yml`, update the h5file path to point to your `.h5` file, and start training by running:

```bash
python train.py --config=./configs/AdaptiveDiffuseMotion.yml
```
## Cite

```
@INPROCEEDINGS{11463428,
  author={Xuan, Enyun and Li, You and Li, Ziwei and Yao, Mengmeng and Guo, Renzhong},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={AdaptiveDiffuseMotion: Adaptive Multi-Task Diffusion Model for Speech-Driven Holistic Motion Generation}, 
  year={2026},
  pages={20057-20061},
  keywords={Motion pictures;MIMICs;Millimeter wave integrated circuits;Monolithic integrated circuits;Protocols;HTTP;Internet;Diversity methods;LoRa;Videos;co-speech motion generation;speech-to-gesture generation;vector quantization;diffusion model},
  doi={10.1109/ICASSP55912.2026.11463428}}
```

## Reference

Our work is based on [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture), [BEAT](https://github.com/PantoMatrix/PantoMatrix)
