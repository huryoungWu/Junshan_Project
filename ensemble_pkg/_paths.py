"""布局无关的路径发现 —— ensemble_pkg 中所有需要 transformer_pkg 的模块从这里走。

军山项目目录结构:
    D:\Junshan_Project\
    ├── transformer_pkg/          # Transformer 模型训练 + 推理
    ├── ensemble_pkg/             # ★ 本融合包
    └── data/                     # 数据文件

"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

# 判定"这个目录是不是一个完整可用的 transformer_pkg"
_REQUIRED_FILES = (
    "data_processing.py",
    "inference_nextday_16h.py",
    "train_transformer_nextday_16h.py",
    "transformer_model.py",
    "itransformer_model.py",
)

CANDIDATES = (
    os.path.join(PROJECT_ROOT, "transformer_pkg"),
)


def find_transformer_pkg():
    """返回第一个包含全部必需文件的 transformer_pkg 目录。"""
    found = [p for p in CANDIDATES
             if all(os.path.isfile(os.path.join(p, f)) for f in _REQUIRED_FILES)]
    if not found:
        raise ImportError(
            "找不到可用的 transformer_pkg。已探测:\n"
            + "\n".join(f"  - {p}" for p in CANDIDATES)
            + f"\n每个目录都需包含: {', '.join(_REQUIRED_FILES)}")
    if len(found) > 1:
        print(f"[paths] [注意] 发现多份 transformer_pkg, 用第一个: {found[0]}")
    return found[0]


def ensure_import_paths(verbose=True):
    """把 transformer_pkg 目录放到 sys.path 最前面 (与 junshan_inference.py 一致)。

    这样 from data_processing import ... 等裸导入就能找到 transformer_pkg 内的模块。
    """
    pkg_dir = find_transformer_pkg()
    norm = os.path.normcase(os.path.abspath(pkg_dir))
    sys.path[:] = [p for p in sys.path
                   if os.path.normcase(os.path.abspath(p or os.getcwd())) != norm]
    sys.path.insert(0, pkg_dir)
    if verbose:
        print(f"[paths] transformer_pkg -> {pkg_dir}")
    return pkg_dir
