Pseudo-Domain-Conditioned Reverse Distillation
for Multi-class Anomaly Detection on Complex
Textured Surfaces

# MVTec AD Texture Evaluation

This script evaluates a unified multi-class anomaly detection model on the five texture categories of MVTec AD:

```text
carpet, grid, leather, tile, wood
```

The evaluator reads the original MVTec AD directory structure directly. It does not require pseudo-domain txt files or a manually merged dataset. If fewer than five texture categories are present under `mvtec_root`, the script evaluates only the existing ones. Non-texture categories are ignored.

## Dataset Structure

Expected MVTec AD layout:

```text
mvtec/
├── carpet/
│   ├── test/
│   └── ground_truth/
├── grid/
│   ├── test/
│   └── ground_truth/
├── leather/
│   ├── test/
│   └── ground_truth/
├── tile/
│   ├── test/
│   └── ground_truth/
└── wood/
    ├── test/
    └── ground_truth/
```

## Command

From the project root:

```bash
python eval_mvtec_texture.py --mvtec_root mvtec --checkpoint .\checkpoints\student_distill_texture_best.pth --image_size 256
```

Command:

```bash
python eval_mvtec_texture.py --mvtec_root mvtec --checkpoint .\checkpoints\student_distill_texture_best.pth --image_size 256
```

Output:

```text
Evaluating categories: carpet, grid, leather, tile, wood

===== Summary =====
Macro Avg: ImageAUROC=0.99780  ImageAP=0.99929  ImageF1max=0.99261  PixelAUROC=0.97909  PixelAP=0.53120  PixelF1max=0.56559  PixelAUPRO=0.94471

===== Per-category Results =====
name    n       I-AUROC I-AP    I-F1max P-AUROC P-AP    P-F1max P-AUPRO
carpet  117     0.99639 0.99897 0.99435 0.99171 0.61508 0.63386 0.96694
grid    78      0.99916 0.99970 0.99130 0.99246 0.48236 0.49638 0.97522
leather 124     1.00000 1.00000 1.00000 0.99478 0.52213 0.51960 0.98408
tile    117     0.99784 0.99913 0.99408 0.95935 0.53164 0.64730 0.86487
wood    79      0.99561 0.99865 0.98333 0.95714 0.50478 0.53079 0.93243
```

## Notes

This checkpoint is intended for the five MVTec AD texture categories only. Categories outside `carpet`, `grid`, `leather`, `tile`, and `wood` are skipped automatically.




