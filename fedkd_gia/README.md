# FedKD under Gradient Inversion Attacks — baseline

Port of [FedKD](https://github.com/wuch15/FedKD) (Wu et al., *Nature Communications* 2022, arXiv:2108.13323)
from news recommendation to image classification, used as the federated setting to evaluate
gradient inversion attacks (GIA).

## Setting

| | |
|---|---|
| Data | [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection) (labeled `imgs/train`, 10 classes, 22,424 images) |
| Clients | 26 — one per driver (`subject` in `driver_imgs_list.csv`), all participate every round |
| Split | per client, stratified 80/20 train/test (seed 42); indices saved in `splits.json` |
| Mentor (teacher, private) | `vit_base_patch16_224` (timm, ImageNet pretrained) |
| Mentee (student, shared) | `vit_small_patch16_224` (timm, ImageNet pretrained) |
| Rounds / local epochs / batch | 30 / 5 / 8 |
| Optimizer | Adam, lr 3e-5, fp16 autocast |
| Input | 224×224, normalized with mean=std=0.5, **no augmentation** |
| Aggregation | unweighted mean of client student updates (as in the official code) |

### Differences from the paper / official code
- **No SVD compression**: clients upload the raw student update (communication cost is not studied here).
  This is also the most favorable view for a GIA attacker.
- **Hidden-state loss across widths**: ViT-S (384-d, 6 heads) vs ViT-B (768-d, 12 heads), both 12 blocks.
  Student hidden states go through a per-layer linear map `W_h` (384→768, local to each client, never shared);
  teacher attention maps are averaged over consecutive head pairs (12→6).
- **Gradient semantics made explicit**: the adaptive weight `1/(CE_t+CE_s)` is detached (in the official
  code gradients also flow through it, which rewards increasing the task loss), and each KL term uses the
  other model's prediction as a detached target (paper Eq. 1–2).
- Embedding parameters (`patch_embed`, `pos_embed`, `cls_token`) are frozen in both models, as in the
  official code (`--freeze_embeddings 1`), so they are not part of the shared update.

Loss per local step:
```
L = CE_t + CE_s + (KL(p_s‖p_t) + KL(p_t‖p_s) + 2·Σ_l [MSE(W_h H_s^l, H_t^l) + MSE(A_s^l, A_t^l)]) / (CE_t + CE_s)
```

## Evaluation (logged every round to `metrics.jsonl`)
- `global_student_acc`, `global_student_macro_f1`: aggregated student on the union of all clients' test splits
- `per_client[*].student_acc`: aggregated student on each driver's own test split
- `per_client[*].teacher_test_acc`: each client's private mentor on its own test split
- train loss components and train accuracy per client

## Running
Local:
```bash
python train.py --data_root /path/to/state-farm --cache_dir cache --out_dir runs/baseline
```
Debug on CPU: add `--max_clients 2 --max_steps 2 --rounds 2 --pretrained 0 --num_workers 0`.

Kaggle (T4×2, clients trained in parallel across GPUs, auto-resume across sessions):
```bash
kaggle kernels push -p fedkd_gia/kaggle/kernels/bench      # short speed test
kaggle kernels push -p fedkd_gia/kaggle/kernels/baseline   # full run
```
To continue after the 12 h session limit, add `"mohamedelsayed10/fedkd-baseline"` to `kernel_sources`
in `kernels/baseline/kernel-metadata.json` and push again; the previous output is detected and resumed.

## CIFAR-100 (IID)

`--split iid` with a non-State-Farm `--data_root` uses CIFAR-100 (downloaded by torchvision):
the 50k training images are dealt class-by-class over `--num_clients 30` clients (IID, ~16-17 images
per class each), every client keeps a stratified 80/20 local train/test split (1,333 / 334 images),
and the official 10k test set is the headline global test. Images stay at native 32x32 in the cache;
the ViTs upsample to 224 internally, so gradient-inversion attacks reconstruct 32x32 images.
Kaggle: `kaggle kernels push -p fedkd_gia/kaggle/kernels/cifar_baseline` (1 local epoch, 30 rounds),
then `kernels/cifar_victim` (victim gradients for clients c00-c05).
