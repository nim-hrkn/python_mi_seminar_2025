import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_df_diff(df_orig: pd.DataFrame, df_recom: pd.DataFrame, threshold: float):
    """plot difference of df

    Args:
        df_orig (pd.DataFrame): original data.
        df_recom (pd.DataFrame): recommended data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax = axes[0]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_orig.values, ax=ax)
    ax.set_title("original")
    ax = axes[1]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_recom.values, ax=ax)
    ax.set_title("low rank approx.")
    ax = axes[2]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_recom.values-df_orig.values > threshold, ax=ax)
    ax.set_title("difference")


def plot_df_heatmap(df, metadata=None, dpi=600):
    """plot headmap of df data

    Args:
        df (pd.DataFrame): data
        metadata (dict, optional): 表示用データ. Defaults to None.
        dpi (float, optional): figure dpi. Defaults to 600.
    """
    size = np.array(df.shape)[::-1]*0.1
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    sns.set(font_scale=0.6)
    sns.heatmap(df, lw=0.1, ax=ax)
    plt.tight_layout()
    if metadata is not None:
        filename = "_".join([metadata["prefix"], metadata["dataname"], "original_heatmap"])+".png"
        print(filename)
        fig.savefig(os.path.join(metadata["outputdir"], filename))


def plot_2df(df, df_transform, nrank, threshold, title_fontsize=30, figsize_factor=1.8, dpi=100, metadata=None):
    """plot the original data and reconstructed data

    Args:
        df (pd.DataFrame): data
        df_transform (pd.DataFrame): reconstructed data
        nrank (int): rank
        threshold (float): 推薦値の差のしきい値.
        title_fontsize (int, optional): title font size. Defaults to 20.
        figsize_factor (float, optional): factor to determine figsize. Defaults to 1.8.
        dpi (float, optional): figure dpi. Defaults to 600.
        metadata (dict, optional): 可視化用データ. Defaults to METADATA.
    """
    figsize = np.array(df.shape).astype(float)[::-1]*0.2
    figsize[0] = figsize[0]*figsize_factor
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    # ax = axes[0]
    # sns.heatmap(df, ax=ax)
    ax = axes[0]
    ax.set_title("nrank={}".format(nrank), fontsize=title_fontsize)
    sns.heatmap(df_transform.values, lw=0.1, ax=ax)
    ax = axes[1]
    ax.set_title("diff".format(nrank),  fontsize=title_fontsize)
    sns.heatmap(df_transform.values-df.values > threshold, lw=0.1, ax=ax)

    fig.tight_layout()
    if metadata is not None:
        filename = "_".join([metadata["prefix"], metadata["dataname"], "lowrank_diff"])+".png"
        print(filename)
        fig.savefig(os.path.join(metadata["outputdir"], filename))

    # fig.show()


def plot_svd_sdiag(X):
    """寄与率の図示を行う．

    Args:
        X (np.array): descriptor
    """
    u, sdiag, v = np.linalg.svd(X)
    if False:
        n = sdiag.shape[0]
        s = np.zeros((u.shape[1], v.shape[0]))
        s[:n, :n] = np.diag(sdiag)
        u = np.matrix(u)
        v = np.matrix(v)  # = v.T
        s = np.matrix(s)
        usv = u*s*v
        print("check usv = original matrix? ", np.allclose(usv, X))

    sdiagsum = []
    for i in range(sdiag.shape[0]):
        sdiagsum.append(np.sum(sdiag[:i+1]))
    sdiagsum = np.array(sdiagsum)
    sdiag = sdiag / sdiagsum[-1]
    sdiagsum = sdiagsum / sdiagsum[-1]

    # 寄与率の表示
    fig, ax = plt.subplots()
    # plt.plot(np.log10(sdiag),"o-")
    ax.plot(sdiag, ".-", label="contribution")
    ax.plot(sdiagsum, ".-", label="comulative contribution")
    ax.set_ylabel("rate")
    ax.set_xlabel("index")
    ax.legend()
    fig.tight_layout()
    # fig.show()
