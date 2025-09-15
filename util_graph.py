import os
import numpy as np
from PIL import Image
from typing import Any, Optional
from scipy.sparse import coo_matrix, dia_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ---- Optional CRF dependency ----
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    _HAS_CRF = True
except Exception:
    _HAS_CRF = False

# ========== GRAPH BUILDING ==========

def _build_grid_edges(H, W, connectivity: int = 8):
    """Build 4- or 8-neighbor grid edges"""
    idx = np.arange(H * W).reshape(H, W)
    src, dst = [], []

    # right
    src.append(idx[:, :-1].ravel()); dst.append(idx[:, 1:].ravel())
    # down
    src.append(idx[:-1, :].ravel()); dst.append(idx[1:, :].ravel())
    if connectivity == 8:
        # diag down-right
        src.append(idx[:-1, :-1].ravel()); dst.append(idx[1:, 1:].ravel())
        # diag up-right
        src.append(idx[1:, :-1].ravel()); dst.append(idx[:-1, 1:].ravel())

    return np.concatenate(src), np.concatenate(dst)

def _edge_weights_rgbxy(image, src, dst, sigma_color=15.0, sigma_space=2.0, connectivity: int = 8):
    H, W, _ = image.shape
    I = image.reshape(-1, 3).astype(np.float32)
    dc = I[src] - I[dst]
    dc2 = np.sum(dc*dc, axis=1)

    # spatial distance: 1 for orthogonal, 2 for diagonal
    rows = np.arange(H*W)//W
    cols = np.arange(H*W)%W
    dr = rows[src]-rows[dst]; dcg = cols[src]-cols[dst]
    ds2 = np.where((dr!=0)&(dcg!=0), 2.0, 1.0).astype(np.float32)

    w = np.exp(-0.5*(dc2/(sigma_color**2) + ds2/(sigma_space**2)))
    return w

def _laplacian_from_weights(n, src, dst, w, normalized=False):
    data = np.concatenate([w,w])
    rows = np.concatenate([src,dst])
    cols = np.concatenate([dst,src])
    W = coo_matrix((data,(rows,cols)), shape=(n,n)).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()

    if not normalized:
        D = dia_matrix((d,0), shape=(n,n))
        L = D-W
        return L,d
    inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d,1e-12))
    Dm12 = dia_matrix((inv_sqrt_d,0), shape=(n,n))
    I = dia_matrix((np.ones(n),0), shape=(n,n))
    L = I - Dm12 @ W @ Dm12
    return L,d

# ========== DATA I/O ==========

def _open_image(path, convert_to):
    if convert_to=="RGB": return np.array(Image.open(path).convert("RGB"))
    if convert_to=="grayscale": return np.array(Image.open(path).convert("L"))
    return np.array(Image.open(path))

def _get_file_names(folder):
    return sorted([f for f in os.listdir(folder) if not f.startswith('.')])

def _load_images(folder_path, folder_name, convert_to):
    image_dir_path = os.path.join(folder_path, folder_name)
    filenames = _get_file_names(image_dir_path)
    return np.stack([_open_image(os.path.join(image_dir_path,f), convert_to) for f in filenames])

def _get_palette(folder_path, ground_truth_dir, filename):
    return Image.open(os.path.join(folder_path, ground_truth_dir, filename)).getpalette()

def _get_filenames(folder_path, scribbles_dir):
    return _get_file_names(os.path.join(folder_path, scribbles_dir))

def load_dataset(folder_path: str, images_dir: str, scribbles_dir: str, ground_truth_dir: Optional[str]=None):
    images = _load_images(folder_path, images_dir, "RGB")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)
    if ground_truth_dir is None:
        return images, scribbles, filenames
    gt = _load_images(folder_path, ground_truth_dir, None)
    palette = _get_palette(folder_path, ground_truth_dir, filenames[0])
    return images, scribbles, gt, filenames, palette

def store_predictions(predictions, folder_path, predictions_dir, filenames, palette):
    os.makedirs(os.path.join(folder_path,predictions_dir), exist_ok=True)
    for fn, pred in zip(filenames, predictions):
        img = Image.fromarray(pred.astype(np.uint8), mode='P')
        img.putpalette(palette)
        img.save(os.path.join(folder_path,predictions_dir,fn))

# ========== HARMONIC SOLVER WITH BACKGROUND WEIGHTING ==========

def _solve_harmonic(L, y_l_weighted, labeled_mask):
    n = L.shape[0]
    idx_all = np.arange(n)
    idx_u = idx_all[~labeled_mask]
    L_csr = L.tocsr()
    L_uu = L_csr[~labeled_mask][:, ~labeled_mask]
    L_ul = L_csr[~labeled_mask][:, labeled_mask]
    rhs = -L_ul @ y_l_weighted

    # ---- add small diagonal to avoid singularity ----
    from scipy.sparse import dia_matrix
    eps = 1e-5
    L_uu = L_uu + eps * dia_matrix((np.ones(L_uu.shape[0]),0), shape=L_uu.shape)

    f_u = spsolve(L_uu, rhs).astype(np.float32)
    f = np.zeros(n, dtype=np.float32)
    f[labeled_mask] = y_l_weighted
    f[~labeled_mask] = f_u
    return np.clip(f, 0.0, 1.0)

def _crf_refine(image_rgb, prob):
    if not _HAS_CRF: return (prob>=0.5).astype(np.uint8)
    H,W = prob.shape
    d = dcrf.DenseCRF2D(W,H,2)
    softmax = np.stack([1.0-prob, prob], axis=0)
    d.setUnaryEnergy(unary_from_softmax(softmax))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=13, rgbim=image_rgb, compat=10)
    Q = d.inference(5)
    return np.argmax(np.array(Q), axis=0).reshape(H,W).astype(np.uint8)

def segment_with_label_propagation(
        image, scribble,
        sigma_color=12, 
        sigma_space=1.0,
        normalized_laplacian=True, 
        connectivity=8,
        use_crf=True, 
        return_prob=False,
        bg_weight=5.0, 
        fg_weight=1.0
    ):
    H,W,C = image.shape; n=H*W
    src,dst = _build_grid_edges(H,W,connectivity)
    w = _edge_weights_rgbxy(image, src,dst, sigma_color, sigma_space, connectivity)
    L,_ = _laplacian_from_weights(n, src,dst,w, normalized=normalized_laplacian)

    y = scribble.reshape(-1)
    labeled_mask = (y != 255)
    if labeled_mask.sum()==0: return np.zeros((H,W),dtype=np.uint8)

    # ---- weighted labels ----
    y_vals = np.zeros_like(y[labeled_mask], dtype=np.float32)
    weights = np.ones_like(y_vals)
    for i,label in enumerate(y[labeled_mask]):
        if label==0:  # background
            y_vals[i]=0.0
            weights[i]=bg_weight
        elif label==1:  # object
            y_vals[i]=1.0
            weights[i]=fg_weight
    y_l_weighted = y_vals * weights

    f = _solve_harmonic(L, y_l_weighted, labeled_mask).reshape(H,W)
    if return_prob: return f
    return _crf_refine(image,f) if use_crf else (f>=0.5).astype(np.uint8)

# ========== VISUALIZATION + METRICS ==========

def _overlay_scribbles(image, scribble, color_fg=(255,0,0), color_bg=(0,0,255), alpha=0.6):
    overlaid=image.copy().astype(np.float32)
    mask_fg = scribble==1; mask_bg=scribble==0
    for mask,color in [(mask_fg,color_fg),(mask_bg,color_bg)]:
        for c in range(3): overlaid[...,c][mask] = alpha*color[c]+(1-alpha)*overlaid[...,c][mask]
    return overlaid.astype(np.uint8)

def visualize(image, scribbles, gt, pred, alpha=0.6):
    image_s = _overlay_scribbles(image, scribbles, alpha=alpha)
    cmap = plt.get_cmap('bwr')
    _, axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(image_s); axes[0].set_title("Image + Scribbles")
    axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=1); axes[1].set_title("Ground Truth")
    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=1); axes[2].set_title("Prediction")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()

def compute_iou(pred, gt, num_classes=2):
    ious=[]
    for c in range(num_classes):
        pc=pred==c; gc=gt==c
        inter = np.logical_and(pc,gc).sum()
        union = np.logical_or(pc,gc).sum()
        ious.append(inter/union if union>0 else float('nan'))
    return ious

def compute_miou(pred_stack, gt_stack, num_classes=2):
    all_ious = np.array([compute_iou(p,g,num_classes) for p,g in zip(pred_stack,gt_stack)])
    mean_per_class = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_per_class)
    return miou, mean_per_class


# ========== SIMPLE GRID SEARCH (optional) ==========

def grid_search_params(images, scribs, gts,
                       sigma_color_list=(8, 10, 12, 15),
                       sigma_space_list=(1.0, 1.5, 2.0),
                       use_crf_list=(True, False),
                       normalized_laplacian_list=(True,),
                       connectivity_list=(8,)):
    """
    Returns best (miou, params, preds)
    """
    best = (-1.0, None, None)
    for sc in sigma_color_list:
        for ss in sigma_space_list:
            for crf_flag in use_crf_list:
                for norm_flag in normalized_laplacian_list:
                    for conn in connectivity_list:
                        preds = []
                        for img, scr in zip(images, scribs):
                            pred = segment_with_label_propagation(
                                img, scr,
                                sigma_color=sc,
                                sigma_space=ss,
                                normalized_laplacian=norm_flag,
                                connectivity=conn,
                                use_crf=crf_flag,
                                return_prob=False
                            )
                            preds.append(pred)
                        preds = np.stack(preds, axis=0)
                        miou, _ = compute_miou(preds, gts)
                        params = dict(sigma_color=sc, sigma_space=ss,
                                      use_crf=crf_flag, normalized=norm_flag, connectivity=conn)
                        if miou > best[0]:
                            best = (miou, params, preds)
    return best
