from tokenize import PlainToken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatterplot_rd(X_rd: np.ndarray, nnatom_str_list: np.ndarray, X_rd_all: np.ndarray = None, alpha=1,
        legendfontsize=15, labelfontsize=15, tickfontsize=15):
    """次元圧縮後の説明変数を図示する。nnatom_str_listをlegendとして使用する。

    Args:
        X_rd (np.ndarray): (N_obs,P) 次元圧縮後の説明変数（ラベルがあるもののみ）
        X_rd_all (np.ndarray): (N_all,P) 次元圧縮後の説明変数（全データ）
        df (pd.DataFrame): (N_obs) nnatom_str
        alpha (float): X_rdの点のalpha値.
        legendfontsize (int, optional): legend font size. Defaults to 15.
        labelfontsize (int, optional): legend font size. Defaults to 15.
        tickfontsize (int, optional): legend font size. Defaults to 15.
    """
    marker_list = ["x", "v", "^", "+", "D", "*"]
    color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
    nnatom_str_uniqlist = np.unique(nnatom_str_list)
    fig, ax = plt.subplots()
    if X_rd_all is not None:
        ax.scatter(X_rd_all[:, 0], X_rd_all[:, 1], s=5, c="black",
                   alpha=0.05)
    for nnatom, marker, color in zip(nnatom_str_uniqlist, marker_list,
    color_list):
        ilist = np.where(nnatom_str_list == nnatom)
        ax.scatter(X_rd[ilist, 0], X_rd[ilist, 1], c=color, 
                   marker=marker, alpha=alpha, label=nnatom)
    # legendを外に出す。
    ax.legend(bbox_to_anchor=(1.05, 1), fontsize=legendfontsize, 
              loc='upper left', borderaxespad=0,)
    ax.set_xlabel("axis1", fontsize=labelfontsize)
    ax.set_ylabel("axis2", fontsize=labelfontsize)
    ax.tick_params(axis = 'x', labelsize =tickfontsize)
    ax.tick_params(axis = 'y', labelsize =tickfontsize)
    fig.tight_layout()

def plot_expratio(indx, explained_variance_ratio, esum):
    """plot explained variance ratio.
    
    Args:
        indx (List[int]): dimension list.
        explained_variance_ratio (List[float]): explained_variance_ratio.
        esum (np.ndarray): explained_variance_ratio sum.
    """
    fig, ax = plt.subplots()
    ax.plot(indx, explained_variance_ratio, "o-",
            label="explained_variance_ratio")
    ax.plot(indx, esum, "o-",
            label="sum(explained_variance_ratio)")
    ax.legend()
