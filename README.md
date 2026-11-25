# Adaptive Kernel Merge and Fusion

Code for **"Adaptive Kernel Merge and Fusion for Multi-Tenant Inference in Embedded GPUs"**, IEEE Embedded Systems Letters, vol. 16, no. 4, pp. 421–424, 2024. DOI: [10.1109/LES.2024.3351753](https://doi.org/10.1109/LES.2024.3351753)

Inference requests for AlexNet, ResNet18 and VGG16 arrive in one queue. The proposed technique looks at a window of 5 queued requests and groups them by model:

- **Kernel merging**: requests of the same model are processed by one kernel launch that iterates over the requests.
- **Kernel fusion**: the merged AlexNet and ResNet18 kernels are fused, so one kernel computes a layer of both models.

This reduces the number of concurrent kernels and the GPU context switches between them. The repository also contains the baselines it is compared with: sequential execution, concurrent execution, and layer-wise interleaved scheduling (ARAS-style prior work).

## Requirements

- NVIDIA Jetson AGX Orin (tested with JetPack 5 / L4T R35.3.1)
- CUDA (tested with 11.4)
- Python 3 with NumPy
- For `export_weights.py` only: PyTorch, torchvision and Pillow (tested with torch 2.0.1, torchvision 0.16.1)

## Repository layout

```
├── exp.sh                          # Build + run all experiments
├── export_weights.py               # Export pretrained weights to data_*/ (see below)
├── experiments_sequential.py       # Sequential execution (baseline)
├── experiments_concurrent.py       # Concurrent execution, one process per model (baseline)
├── experiments_aras.py             # Layer-wise interleaved scheduling (prior work)
├── experiments_merge_fusion.py     # Adaptive kernel merge and fusion (proposed)
├── parsing.py                      # Average queuing delay / inference time / latency of a result file
├── random_model.txt                # Random order of 2000 requests (AlexNet / ResNet18 / VGG16)
├── make_random_model.py            # Script that generated random_model.txt
├── imagenet1000_clsidx_to_labels.txt  # ImageNet class names (used to print the predicted class)
├── images/cat.jpg                  # Input image
└── src/
    ├── kernels.cu                  # Basic kernels (conv, pooling, LRN, batch norm, fc, ...)
    ├── fused_kernels.cu            # Merged kernels (*_merged) and fused AlexNet+ResNet18 kernels (fused_*)
    ├── basic_models.cu             # Whole-model inference, one request per call      -> basic_models.so
    ├── aras_models.cu              # One host function per layer                      -> aras_models.so
    └── merge_fusion.cu             # Merged/fused AlexNet+ResNet18, merged VGG16      -> merge_fusion.so
```

| Policy | Script | Shared library |
|---|---|---|
| Sequential | `experiments_sequential.py` | `aras_models.so` |
| Concurrent | `experiments_concurrent.py` | `basic_models.so` |
| ARAS (prior work) | `experiments_aras.py` | `aras_models.so` |
| **Merge and Fusion** | `experiments_merge_fusion.py` | `merge_fusion.so` |

## Model weights

The models use the pretrained ImageNet weights from torchvision (`IMAGENET1K_V1`), stored as text files. They are not in the repository because they take about 2.9GB. Create them with:

```bash
./export_weights.py              # all models; or e.g. ./export_weights.py alexnet
```

This downloads the torchvision weights and writes:

```
data_alexnet/    conv1-5.txt, fc6-8.txt, bias1-8.txt, input_cat1.txt                 (~830MB)
data_resnet18/   conv_data/ gamma_data/ beta_data/ mean_data/ var_data/ fc_data/,
                 input_cat.txt                                                        (~160MB)
data_vgg16/      conv1-13.txt, fc14-16.txt, bias1-16.txt, input_cat.txt               (~1.9GB)
```

Each file holds one value per line. Conv weights are `[out][in][kh][kw]`, FC weights are `[out][in]`, and the input is `images/cat.jpg` resized to 224×224, normalized with the ImageNet mean/std, in HWC order. When run in a terminal, the warm-up inferences print the predicted class (`Egyptian cat` for AlexNet, `tabby cat` for ResNet18 and VGG16), which shows that the weights are loaded correctly.

## Build and run

Run everything from the repository root, because the scripts use relative paths.

```bash
# Build the shared libraries used by the Python scripts
nvcc -Xcompiler -fPIC -shared -o basic_models.so src/basic_models.cu
nvcc -Xcompiler -fPIC -shared -o aras_models.so  src/aras_models.cu
nvcc -Xcompiler -fPIC -shared -o merge_fusion.so src/merge_fusion.cu

# Run all policies at all arrival rates (1000 requests each)
./exp.sh
```

Single runs:

```bash
mkdir -p results
./experiments_sequential.py   <num_requests> <arrival_rate>
./experiments_concurrent.py   <num_requests> <arrival_rate>
./experiments_aras.py         <num_requests> <arrival_rate>
./experiments_merge_fusion.py <num_requests> <arrival_rate>
```

- `<num_requests>`: number of requests to process.
- `<arrival_rate>`: `95`, `90`, `85` or `80`. Requests arrive as a Poisson process whose rate is that percentage of the service rate. The service rate comes from `avg_inference_time` in each script (0.0762 s, the measured average inference time of the three models). The request queue is limited to about ρ²/(1−ρ) requests (ρ = arrival rate / service rate, i.e., the mean queue length of an M/M/1 queue); requests that arrive when it is full are dropped. For merge and fusion the limit is at least 5, the size of the merge window.

The model of the *i*-th request is the *i*-th line of `random_model.txt`, so every policy sees the same sequence of models.

## Results

Each run appends one line per request to `results/<policy>.txt`:

```
<model>  <queuing_delay>  <inference_time>      (seconds)
```

For merge and fusion, `<model>` shows the merged group, e.g. `Fused model(Alex:2 ,Res:3)` or `Fused model(Vgg:2)`, and `<inference_time>` is the group's execution time divided by the number of requests in it. The script also prints the total time and the number of processed requests.

Result files are opened in append mode. Move old results away before running again (`exp.sh` renames them to `results/<policy>_<rate>.txt`).

```bash
./parsing.py results/merge_fusion.txt   # requests, avg queuing delay, avg inference time, avg / 99th-pct latency
```

## Citation

```bibtex
@article{jeon2024mergefusion,
  title   = {Adaptive Kernel Merge and Fusion for Multi-Tenant Inference in Embedded GPUs},
  author  = {Jeon, Jaebeom and Koo, Gunjae and Yoon, Myung Kuk and Oh, Yunho},
  journal = {IEEE Embedded Systems Letters},
  volume  = {16},
  number  = {4},
  pages   = {421--424},
  year    = {2024},
  doi     = {10.1109/LES.2024.3351753}
}
```
