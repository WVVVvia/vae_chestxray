# VAE on Chest X-Rays

A convolutional VAE written from scratch in PyTorch. It is trained on the
[Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and
evaluated with FID and Inception Score using PyTorch-Ignite.

`vae_chestxray.ipynb` contains all the code. It is saved with its outputs (training log, metrics, figures), so you
can see the results without running it.

## How to run (Kaggle)

1. Import `vae_chestxray.ipynb` into a new Kaggle notebook.
2. Add the `chest-xray-pneumonia` dataset as input.
3. Set the accelerator to GPU and turn Internet on (needed to download the Inception-v3 weights).
4. Run all cells. The whole run takes about 5 minutes, and outputs are saved to `/kaggle/working/outputs/`.

## Results

| | FID (2048-d) ↓ | FID (ignite default) ↓ | IS ↑ |
|---|---|---|---|
| VAE samples | 133.77 | 0.0395 | 1.494 |
| Real images (reference) | 4.47 | 0.0008 | 1.956 |

Each row uses 2,500 images and is compared against 2,500 training images. The "real images" row scores a separate
set of real X-rays, which gives the best value this setup can reach.
