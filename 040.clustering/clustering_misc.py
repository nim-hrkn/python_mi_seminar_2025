import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


import seaborn as sns


def plot_X_withlabel(X_PCA, X_new_PCA, ans_list,
                     figsize=(6, 5), alpha_kde=0.3, alpha_scatter=0.1, filename=None):
    """plot X_PCA, X_new_PCA with ans_list

    Args:
        X_PCA (np.ndarray): PCA transformed X.
        X_new_PCA (np.ndarray): PCA transformed X_new.
        ans_list (np.ndarray): answer of X_new.
        figsize (Tuple(float)): figure size. Defaults to (5,5).
        filename (str): filename if not None. Defaults to None.
    """
    col_list = ["red", "blue", "black", "green", "purple", "yellow"]
    col_list = ["black"]*6
    marker_list = ["s", "o", "^", "x", ".", "v"]
    uniq_ans = np.unique(ans_list)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.kdeplot(x=X_PCA[:, 0], y=X_PCA[:, 1], alpha=alpha_kde)
    ax.scatter(X_PCA[:, 0], X_PCA[:, 1], marker=".", alpha=alpha_scatter)
    for i, (ans, col, marker) in enumerate(zip(uniq_ans, col_list, marker_list)):
        ilist = np.where(ans_list == ans)
        ax.scatter(X_new_PCA[ilist, 0], X_new_PCA[ilist, 1], c=col,
                   marker=marker, label=ans)
    ax.set_aspect("equal", "box")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    plt.show()

def plot_X_clusters(X_PCA, yp_km, alpha=0.1, tickfontsize=15, legendfontsize=15, filename=None):
    """plot X clusters

    Args:
        X_PCA (np.ndarray): PCA transformed X.
        yp_km (np.ndarray): k-Means cluster numbers.
        alpha (float): alpha value of points. Defaults to 0.1.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): ticks font size. Defaults to 15.
        filename (str): filename if not None. Defaults to None.
    """
    figsize = (5, 10)
    uniq_yp = np.unique(yp_km)
    xlim = (X_PCA[:, 0].min(), X_PCA[:, 0].max())
    ylim = (X_PCA[:, 1].min(), X_PCA[:, 1].max())

    marker_list = ["s", "o", "^", "x", ".", "v"]
    col_list = ["black"]*6
    figsize = (5, 5)
    fig, axes = plt.subplots(uniq_yp.shape[0], 1, figsize=figsize)
    for i, (yp, col, marker) in enumerate(zip(uniq_yp, col_list, marker_list)):
        ax = axes[i]
        ax.set_aspect("equal", "box")
        ilist = np.where(yp_km == yp)
        ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                   marker=marker, label=yp, alpha=alpha)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legendfontsize)
        if i != axes.shape[0]-1:
            ax.xaxis.set_ticklabels([])
        ax.tick_params(axis='x', labelsize=tickfontsize)
        ax.tick_params(axis='y', labelsize=tickfontsize)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    plt.show()

def plot_X2_ystring(X_PCA, yp_km, ans_list,  X_new=None, alpha=0.01, figsize=(6, 4),
                    comment=None, metadata=None):
    """plot X2 and label.

    Args:
        X_PCA (np.ndarray): X.
        yp_km (np.ndarray): predicted integer labels.
        ans_list (np.ndarray): observed string labels.
        X_new (np.ndarray): X new. Defaults to None.
        alpha (float): alpha value of plot. Defaults to 0.01.
        figsize (Tuple[float,float]): figure size. Defaults to (5,10).
        comment (str): comment for png output. Defaults to None.
        metadata (dict): data for png output. Defaults to None.
    """
    uniq_yp = np.unique(yp_km)
    uniq_ans = np.unique(ans_list)

    col_list = ["red", "blue", "black", "green", "purple", "yellow"]
    col_list = ["black"]*6
    marker_list = ["s", "o", "^", "x", ".", "v"]
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    if True:
        ax = axes[1]
        ax.set_aspect("equal", "box")
        if X_new is not None:
            ax.scatter(X_new[:, 0], X_new[:, 1], marker=".", alpha=alpha, c="black")
        for y, col, marker in zip(uniq_ans, col_list, marker_list):
            ilist = np.where(ans_list == y)
            ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                       marker=marker, label=y)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if True:
        ax = axes[0]
        ax.set_aspect("equal", "box")
        if X_new is not None:
            ax.scatter(X_new[:, 0], X_new[:, 1], marker=".", alpha=alpha, c="black")
        for i, (y, col, marker) in enumerate(zip(uniq_yp, col_list, marker_list)):
            ilist = np.where(yp_km == y)
            ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                       marker=marker, label=y)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    if metadata is not None:
        filename = "_".join([metadata["prefix"], comment])+".png"
        print(filename)
        plt.savefig(os.path.join(metadata["outputdir"], filename))
    plt.show()


def plot_X2_ystring_new(X_PCA, yp_km, ans_list, X_new_PCA=None, yp_km_new=None, figsize=(5, 9)):
    """plot X2 and label.

    Args:
        X_PCA (np.ndarray): X.
        yp_km (np.ndarray): predicted integer labels.
        ans_list (np.ndarray): observed string labels.
        X_new_PCA (np.ndarray): new X. Defaults to None.
        yp_km_new (nd.ndarray): predicted integer labels. Defaults to None.
        figsize (Tuple[float,float]): figure size. Defaults to (5,10).
    """
    uniq_yp = np.unique(yp_km)
    uniq_ans = np.unique(ans_list)

    if X_new_PCA is not None:
        xall = np.vstack([X_PCA, X_new_PCA])
        xlim = (xall[:, 0].min(), xall[:, 0].max())
        ylim = (xall[:, 1].min(), xall[:, 1].max())
    else:
        xlim = None
        ylim = None

    col_list = ["red", "blue", "black", "green", "purple", "yellow"]
    col_list = ["black"]*6
    marker_list = ["s", "o", "^", "x", ".", "v"]
    fig, axes = plt.subplots(1+uniq_yp.shape[0], 1, figsize=figsize)

    if True:
        ax = axes[0]
        ax.set_aspect("equal", "box")
        for y, col, marker in zip(uniq_ans, col_list, marker_list):
            if False:
                if yp_km_new is not None:
                    ilistnew = np.where(yp_km_new == y)
                    ax.scatter(X_new_PCA[ilistnew, 0], X_new_PCA[ilistnew, 1], c=col,
                               marker=marker, alpha=0.05)
            ilist = np.where(ans_list == y)
            ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                       marker=marker, label=y)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if True:

        # ax.set_aspect("equal", "box")
        for i, (y, col, marker) in enumerate(zip(uniq_yp, col_list, marker_list)):
            ax = axes[1+i]
            ax.set_aspect("equal", "box")
            if yp_km_new is not None:
                ilistnew = np.where(yp_km_new == y)
                ax.scatter(X_new_PCA[ilistnew, 0], X_new_PCA[ilistnew, 1], c=col,
                           marker=marker, alpha=0.05)
            ilist = np.where(yp_km == y)
            ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                       marker=marker, label=y)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    plt.show()


def plot_X2_new(X_PCA, yp_km, X_new_PCA=None, yp_km_new=None, figsize=(5, 9)):
    """plot X2 and label.

    Args:
        X_PCA (np.ndarray): X.
        yp_km (np.ndarray): predicted integer labels.
        X_new_PCA (np.ndarray): new X. Defaults to None.
        yp_km_new (nd.ndarray): predicted integer labels. Defaults to None.
        figsize (Tuple[float,float]): figure size. Defaults to (5,10).
    """
    uniq_yp = np.unique(yp_km)

    if X_new_PCA is not None:
        xall = np.vstack([X_PCA, X_new_PCA])
        xlim = (xall[:, 0].min(), xall[:, 0].max())
        ylim = (xall[:, 1].min(), xall[:, 1].max())
    else:
        xlim = None
        ylim = None

    col_list = ["red", "blue", "black", "green", "purple", "yellow"]
    col_list = ["black"]*6
    marker_list = ["s", "o", "^", "x", ".", "v"]
    fig, axes = plt.subplots(uniq_yp.shape[0], 1, figsize=figsize)

    if True:

        # ax.set_aspect("equal", "box")
        for i, (y, col, marker) in enumerate(zip(uniq_yp, col_list, marker_list)):
            ax = axes[i]
            ax.set_aspect("equal", "box")
            if yp_km_new is not None:
                ilistnew = np.where(yp_km_new == y)
                ax.scatter(X_new_PCA[ilistnew, 0], X_new_PCA[ilistnew, 1], c=col,
                           marker=marker, alpha=0.05)
            ilist = np.where(yp_km == y)
            ax.scatter(X_PCA[ilist, 0], X_PCA[ilist, 1], c=col,
                       marker=marker, label=y)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    plt.show()


def make_df_sample(df, descriptor_names, n, group_name="nnatom_str", random_state=3):
    """sample df and make X and ans.
    Choose n samples of each groups

    Args:
        df (pd.DataFrame): data.
        descriptor_names (List[str]): a list of explanatory vairables.
        n (int): number of samples.
        group_name (str) : groupby name. Defaults to "nnatom_str".
        random_state (int, optional): random state for df.sample(). Defaults to 3.

    Returns:
        _type_: _description_
    """
    df_sample = df.groupby(group_name).sample(n, random_state=random_state)
    if False:
        name_list = []
        for ans, _x in zip(df_sample[group_name], df_sample.index.tolist()):
            _x = list(map(str, _x))
            s = "_".join([ans, _x[0], _x[1]])
            name_list.append(s)
        df_sample["name"] = name_list
    Xraw_sample = df_sample[descriptor_names].values
    scaler_sample = StandardScaler()
    scaler_sample.fit(Xraw_sample)
    X_sample = scaler_sample.transform(Xraw_sample)
    ans_list_sample = df_sample[group_name].values
    return X_sample, ans_list_sample
