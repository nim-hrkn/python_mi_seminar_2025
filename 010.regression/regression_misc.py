import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_alpha_yerror(
    df: pd.DataFrame,
    show_detail: bool = False,
    tickfontsize=15,
    labelfontsize=15,
    legendfontsize=15,
):
    """alphaの変化を図示する．

    Args:
        df (pd.DataFrame): 回帰スコアの平均値と標準偏差とalpha
        show_detail (bool, optiona): 0.99-1.0を拡大表示するか．Defaults to False.
        tickfontsize (int, optional): tick font size. Defaults to 15.
        labelfontsize (int, optional): tick font size. Defaults to 15.
        legendfontsize (int, optional): tick font size. Defaults to 15.
    """
    alpha_list = df["alpha"]

    fig, ax = plt.subplots()

    if "mean(R2)_test" in df.columns.tolist():
        mean_score_list = df["mean(R2)_test"]
        std_score_list = df["std(R2)_test"]
        ax.errorbar(
            np.log10(alpha_list),
            mean_score_list,
            yerr=std_score_list,
            fmt="o-",
            capsize=5,
            label="test",
        )

    if "mean(R2)_train" in df.columns.tolist():
        mean_score_list = df["mean(R2)_train"]
        std_score_list = df["std(R2)_train"]
        ax.errorbar(
            np.log10(alpha_list),
            mean_score_list,
            yerr=std_score_list,
            fmt="o-",
            capsize=5,
            label="train",
        )

    ax.set_xlabel("log10(alpha)", fontsize=labelfontsize)
    ax.set_ylabel("$R^2$", fontsize=labelfontsize)
    ax.legend(fontsize=legendfontsize)
    if show_detail:
        # 0.99以上のみ表示する．
        plt.ylim((0.9, 1.01))
        # plt.savefig("image_executed/alpha_vs_R2.png")
    ax.tick_params(axis="x", labelsize=tickfontsize)
    ax.tick_params(axis="y", labelsize=tickfontsize)
    fig.tight_layout()
    plt.show()


def plot_y_yp(y, yp, title: str = None, tickfontsize=15, labelfontsize=15):
    """y vs ypを図示する．

    Args:
        y (np.ndarray): 目的変数値
        yp (np.ndarray|list): 目的変数予測値, 1D or 2D.
        title (str, optional): 図のtitle. Defaults to None.
        tickfontsize (int, optional): tick font size. Defaults to 15.
        labelfontsize (int, optional): tick font size. Defaults to 15.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # $y^{obs}$ vs $y^{predict}$
    for y1, yp1 in zip(y, yp):
        ax.plot(y1, yp1, "o")

    # 斜め線を引く
    # check y is 1 dim or 2dim
    if isinstance(y,np.ndarray):
        if len(y[0].shape)>0:
            y = np.concatenate(y)
            yp = np.concatenate(yp)
        
    yall = np.hstack([y,yp])
    ylim = yall.min(), yall.max()
    ax.plot(ylim, ylim, "--")

    # labelを書く
    ax.set_xlabel("$y_{obs}$", fontsize=labelfontsize)
    ax.set_ylabel("$y_{pred}$", fontsize=labelfontsize)
    if title is not None:
        ax.title(title)
    ax.tick_params(axis="x", labelsize=tickfontsize)
    ax.tick_params(axis="y", labelsize=tickfontsize)
    fig.tight_layout()
    plt.show()

def plot_x1_y(X, y, yp, Xnew, ynew, ynewp):
    """plot x1 vs y

    Args:
        X (np.ndarray): X obs.
        y (np.ndarray): y obs.
        yp (np.ndarray): y obs predicted.
        Xnew (np.ndarray): X new.
        ynew (np.ndarray): y new.
        ynewp (np.ndarray): y new predicted.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(X[:, 0], y, ".")
    ax.plot(Xnew[:, 0], ynew, "o", label="$y^{obs}_{new}$")
    ax.plot(X[:, 0], yp, ".")
    ax.plot(Xnew[:, 0], ynewp, "x", label="$y^{pred}_{new}$")
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()
