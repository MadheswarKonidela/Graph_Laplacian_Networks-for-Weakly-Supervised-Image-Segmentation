
# Graph Laplacian Networks for Weakly-Supervised Image Segmentation

 **Achieves 0.70 mIoU using only sparse scribble annotations**
 **Graph-based learning > CNNs under weak supervision**
 **No heavy training required (non-parametric approach)**

---

## Problem

Accurate image segmentation typically requires **dense pixel-level annotations**, which are expensive and time-consuming.

This project solves segmentation using only **scribble annotations** (weak supervision), making it far more practical for real-world applications.

---

## Solution

We model the image as a **graph** and propagate labels using a **Graph Laplacian framework**.

* Pixels → Nodes
* Similarity (color + spatial) → Edge weights
* Scribbles → Seed labels
* Output → Smooth, globally consistent segmentation

✔ Solved via **harmonic energy minimization**
✔ Enhanced with **DenseCRF for sharp boundaries**

---

## Results

| Model                      | mIoU     |
| -------------------------- | -------- |
| KNN Baseline               | 0.65     |
| U-Net (weak supervision)   | 0.52     |
| **Graph Laplacian (Ours)** | **0.70** |

 **Best performing method under weak supervision**

---

## Key Achievements

* **+18% improvement over CNN**
* Strong performance with **minimal annotations**
* Preserves **object boundaries better than KNN**
* **Zero training cost** (no backprop, no GPUs required)
* Demonstrates power of **graph-based learning**

---

## Why It Stands Out

* Works well with **limited labeled data**
* Combines **global consistency + local smoothness**
* More reliable than CNNs when supervision is sparse
* Simple, interpretable, and efficient

---

## Limitations

* Sensitive to hyperparameters (σ values)
* Struggles with complex textures / multiple objects
* Outperformed by deep models under full supervision

---

## Tech Stack

`Python` • `NumPy` • `SciPy` • `OpenCV` • `DenseCRF`

---

## Future Improvements

* Hybrid **Graph + Deep Learning models**
* Use **learned feature embeddings**
* Extend to **multi-class segmentation**
* Explore **transformer-based integration**
---

## References

* Grady (2006) – Random Walks for Segmentation
* Xu et al. (2015) – Deep GrabCut
* Tang et al. (2018) – Weakly-Supervised CNNs
