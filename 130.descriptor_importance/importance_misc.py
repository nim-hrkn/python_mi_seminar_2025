
import os
import numpy as np
from tkinter import NONE
import matplotlib.pyplot as plt


def plot_importance(df, x, y, sortkey=None, yscale="log",
                    tickfontsize=15, labelfontsize=15, legendfontsize=15):
    """plot importance.

    Args:
        df (pd.DataFrame): データ.
        x (str): x to plot bar.
        y (str): y to plot bar.
        sortkey (str, optional): sortkey. Defaults to None.
        yscale (str, optional): yscale string. Defaults to "log".
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
        legendfontsize (int, optional): label font size. Defaults to 15.
    """
    fig, ax = plt.subplots()
    if sortkey is None:
        sortkey = y
    _df = df.sort_values(by=sortkey, ascending=False)
    _df.plot.bar(x=x, y=y, ax=ax)
    ax.set_yscale(yscale)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=labelfontsize)
    ax.legend(fontsize=legendfontsize)
    fig.tight_layout()


def show_r2_decrease(df, comment: str = "perm_importance_boxplot", metadata: dict = None,
        tickfontsize=15, labelfontsize=15, legendfontsize=15):
    """R2の減少値を図示する．

    Args:
        df (pd.DataFrame): データ
        metadata (dict): 表示用データ. Defaults to METADATA.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
        legendfontsize (int, optional): label font size. Defaults to 15.        
    """
    score_mean = np.mean(df.values, axis=0)
    iorder = np.argsort(score_mean)[::-1]
    fig, ax = plt.subplots()
    df.iloc[:, iorder].boxplot(rot=90, ax=ax)
    ax.set_ylabel("$R^2$ decrease", fontsize=labelfontsize)
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)    
    fig.tight_layout()
    if metadata is not None:
        filename = "_".join([metadata["prefix"], metadata["dataname"], metadata["regtype"], comment])+".pdf"
        print(filename)
        fig.savefig(os.path.join(metadata["outputdir"], filename))
