import matplotlib.pyplot as plt
import os

try:
    from PIL import Image
    have_PIL = True
except ModuleNotFoundError:
    have_PIL = False


def plot_GPR(X, y, Xtrain, ytrain, y_mean, y_std, acq, it, ia=None, metadata=None, 
            tickfontsize=20, labelfontsize=20, legendfontsize=20, titlefontsize=20,
            figsize=(15,4)):
    """plot y.mean += y.std and aquisition functions

    Args:
        X (np.array): descriptor
        y (np.array): target values
        Xtrain (np.array): training descriptor data 
        ytrain (np.array): training target values
        yp_mean (np.array): the mean values of predictions
        yp_std (np.array): the stddev vlaues of predictions
        acq (np.array): aquisition function values
        ia (np.array, optional): a list of actions. Defaults to None.
        metadata (dict): 表示用データ. Defaults. to METADATA.
        tickfontsize (int. optional): ticks font size. Defaults to 20.
        labelfontsize (int, optional): label font size. Defaults to 20.
        legendfontsize (int, optional): legend font size. Defauls to 20.
        titlefontsize (int, optional): title font size. Defauls to 25.    
        figsize (Tuple[float], optional): figure size. Defaults to (20,30).
    """
    xlabel = "x1"
    if metadata is not None:
        dataname = metadata["dataname"]
        acqname = metadata["acq"]
        if "xlabel" in metadata:
            xlabel = metadata["xlabel"]

    yminus = y_mean - y_std
    yplus = y_mean + y_std
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(Xtrain, ytrain, "o", color="blue", label="train")
    ax1.fill_between(X, yminus, yplus, color="red", alpha=0.2)
    ax1.plot(X, y_mean, color="red", label="predict$\pm\sigma$", linewidth=0.2)
    ax1.plot(X, y, "--", color="blue", label="expriment")
    ax1.set_xlabel(xlabel, fontsize=labelfontsize)
    ax1.legend(fontsize=legendfontsize, bbox_to_anchor=(-.15, 1), loc='upper right',)
    ax1.tick_params(axis = 'x', labelsize =tickfontsize)
    ax1.tick_params(axis = 'y', labelsize =tickfontsize)  
    
    ax2.plot(X, acq, color="green", label="acquisition function", linewidth=1)
    if ia is not None:
        # ax2.axvline(ia,color="green",linestyle="--")
        plt.plot(X[ia], acq[ia], "o", color="purple", label="selected action")
    ax2.set_xlabel(xlabel, fontsize=labelfontsize)
    ax2.legend(fontsize=legendfontsize, bbox_to_anchor=(1.05, 1), loc='upper left',)
    ax2.tick_params(axis = 'x', labelsize =tickfontsize)
    ax2.tick_params(axis = 'y', labelsize =tickfontsize)      
    fig.suptitle("iteration {}".format(it+1), fontsize=titlefontsize)
    
    fig.tight_layout()

    if metadata is not None:
        filename = "{}_BayseOpt_acq_{}_{}.png".format(dataname, acqname, it)
        print(filename)
        fig.patch.set_alpha(1) # 背景は透過でない白色(default色）にする。
        fig.savefig(os.path.join(metadata["outputdir"],filename))
    # fig.show()


if have_PIL:
    def make_acq_animation(nselect, train, metadata):
        """複数獲得関数図からアニメーションを作成する．

        Args:
            nselect (int): 選択開始index
            train (list[int])): actionリスト（観測済データリスト）
            metadata (dict): 表示用データ. Defaults to METADATA.

        Returns:
            _type_: _description_
        """
        outputdir = metadata["outputdir"]
        dataname = metadata["dataname"]
        acq = metadata["acq"]
        imglist = []
        for idx in range(nselect, len(train)):
            idx = idx-nselect
            filename = os.path.join(outputdir,"{}_BayseOpt_acq_{}_{}.png".format(dataname, acq, idx))
            print(filename)
            imglist.append(Image.open(filename))
        # 最初のimageを用いてsaveするという仕様．
        filename_fig = os.path.join(outputdir,"{}_BayseOpt_acq.gif".format(dataname))
        imglist[0].save(filename_fig,
                        save_all=True,
                        append_images=imglist[1:], duration=500,
                        interlace=False,
                        loop=1)
        print("saved to", filename_fig)
        return filename_fig

from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap


def carbon_vis_parameter(df):
    """carbonを選択した場合の”diamond", "graphite"の図示パラメタを得る．

    Args:
        df (pd.DataFrame): データ

    Returns:
        tuplex containing

        - list[int]: ”diamond", "graphite"のindex
        - list[str]: ”diamond", "graphite"のpolytype名
        - list[str]: ”diamond", "graphite"の表示シンボル
    """
    ilist = [0, 1]
    ilabellist = ["diamond", "graphite"]
    ilabellist = [df.loc[0, "polytype"], df.loc[1, "polytype"]]
    imarkerlist = ["s", "^"]
    return ilist, ilabellist, imarkerlist


def show_energysurface(X, y, df, metadata):
    """show target value heatmap in the 2D descriptor space

    Args:
        X (np.array): descriptor
        y (np.array): target values
        df (pd.DataFrame): データ．carbonの場合にpolytypeを得る．
        metadata (dict): 表示用データ. 
    """
    sample = metadata["dataname"]
    pca = PCA(2)
    pca.fit(X)
    X2d = pca.transform(X)

    fig, ax = plt.subplots()
    cm = plt.get_cmap("rainbow")
    im = ax.scatter(X2d[:, 0], X2d[:, 1], marker=".", c=y, cmap=cm, alpha=1)
    fig.colorbar(im, ax=ax)

    if sample == "carbon":
        ilist, ilabellist, imarkerlist = carbon_vis_parameter(df)

        for il, ilabel, imark in zip(ilist, ilabellist, imarkerlist):
            ax.scatter(X2d[il, 0], X2d[il, 1], marker=imark, c="black",
                       label=ilabel)
    else:
        il = 0
        ax.scatter(X2d[il, 0], X2d[il, 1], marker="o", c="black",
                   label="TOP")

    ax.legend()
    fig.show()

def show_2D_actions(X, y, nselect, train, df, sample, metadata=None,
                    show_text = False, 
                    textfontsize=15, tickfontsize=15, legendfontsize=15, titlefontsize=15):
    """2Dに探索過程の可視化をする．

    Args:
        X (np.ndarray): 全X
        y (np.ndarray): 全y
        nselect (int): 選択開始index
        train (list[int]): 観測済データindexリスト
        df (pd.DataFrame): データ
        sample (str): データ名.
        metadata (dict, optional): data for display. Defaults to None.
        show_text (bool, optional): show text in search map or not. Defaults to False.
        textfontsize (int, optional): text font size. Defaults to 15.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): legend font size. Defaults to 15.
        titlefontsize (int, optional): title font size. Defaults to 15.
    """
    pca = PCA(2)
    pca.fit(X)
    X2d = pca.transform(X)

    if sample == "carbon":
        ilist, ilabellist, imarkerlist = carbon_vis_parameter(df)

    for idx in range(nselect, len(train)):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("iteration {}".format(idx-nselect+1), fontsize=titlefontsize)
        cm = plt.get_cmap("rainbow")
        im = ax.scatter(X2d[:, 0], X2d[:, 1], marker=".",
                        c="gray", cmap=cm, alpha=0.1)
        fig.colorbar(im, ax=ax)
        # 選択された点を大きく描く
        ax.scatter(X2d[train[:idx-1], 0], X2d[train[:idx-1], 1],
                   marker="x", c=y[train[:idx-1]], alpha=1,
                   vmin=y.min(), vmax=y.max(), cmap=cm)

        if sample == "carbon":
            # [0,1]を見つける問題なので大きく描く
            for il, ilabel, imark in zip(ilist, ilabellist, imarkerlist):
                ax.scatter(X2d[il, 0], X2d[il, 1], marker=imark, c="black",
                           label=ilabel)
        else:
            ax.scatter(X2d[0, 0], X2d[0, 1], marker="o", c="black",
                       label="TOP")

        i1 = train[idx-1]
        i2 = train[idx]
        if show_text:
            ax.text(X2d[i1, 0], X2d[i1, 1], str(i1), fontsize=textfontsize)
            ax.text(X2d[i2, 0], X2d[i2, 1], str(i2), fontsize=textfontsize)
        #  i1 -> i2の矢印を引く
        ax.arrow(X2d[i1, 0], X2d[i1, 1],
                 X2d[i2, 0] - X2d[i1, 0], X2d[i2, 1] - X2d[i1, 1],
                 width=0.05, head_width=0.5,
                 length_includes_head=True, color="black", alpha=1)
        # ax.legend(fontsize=legendfontsize)
        ax.tick_params(axis = 'x', labelsize =tickfontsize)
        ax.tick_params(axis = 'y', labelsize =tickfontsize)  
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.tight_layout()
        if metadata is not None:
            dataname = metadata["dataname"]
            acqname = metadata["acq"]
            filename = os.path.join(metadata["outputdir"], 
                                    "{}_BayseOpt_{}_PCA_{}.png".format(dataname, acqname, idx))
            print(filename)
            fig.patch.set_alpha(1) # 背景は透明ではない。
            fig.savefig(filename)
        #fig.show()


if have_PIL:
    def make_pca_gif(train, nselect, metadata):
        """ベイズ最適化action過程のpng図からgifを作成する．

        Args:
            train (list[int]): action list (観測済データ).
            nselect (int): 選択数. 
            metadata (dict): 補助データ.
        Returns:
            str: gif filename
        """
        dataname = metadata["dataname"]
        acqname = metadata["acq"]
        imglist = []
        for idx in range(nselect, len(train)):
            filename = os.path.join(metadata["outputdir"], 
                                    "{}_BayseOpt_{}_PCA_{}.png".format(dataname, acqname, idx))
            imglist.append(Image.open(filename))
        filename_gif = os.path.join(metadata["outputdir"],
                                    "{}_BayseOpt_{}_PCA.gif".format(dataname, acqname))
        imglist[0].save(filename_gif,
                        save_all=True,
                        append_images=imglist[1:], duration=300,
                        interlace=False,
                        loop=1)
        return filename_gif
