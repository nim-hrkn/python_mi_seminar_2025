import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_index_r2(df_result, y, yerr, xlabel, ylabel, labelfontsize=15,
                  tickfontsize=15, legendfontsize=15, figsize=(10, 5)):
    """plot index vs r2.

    Args:
        df_result (pd.DataFrame): data.
        y (str): y of data.
        yerr (str): yerr of data.
        xlabel (str): string to pass ax.xlabel.
        ylabel (str): string to pass ax.ylbael.
        labelfontsize (int, optional): label font size. Defaults to 15.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): legend font size. Defaults to 15.
        figsize (tuple, optional): figure size. Defaults to (10,5).
    """
    fig, ax = plt.subplots(figsize=figsize)
    df_result.plot(y=y, yerr=yerr, ax=ax)
    ax.set_xlabel(xlabel, fontsize=labelfontsize)
    ax.set_ylabel(ylabel, fontsize=labelfontsize)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.legend(fontsize=legendfontsize)
    fig.tight_layout()
    plt.show()


def plot_importance(df, x, y, sortkey=None, yscale="log",
                    tickfontsize=15, labelfontsize=15, legendfontsize=15, figsize=(10, 5)):
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
        figsize (tuple, optional): figure size. Defaults to (10,5).
    """
    fig, ax = plt.subplots(figsize=figsize)
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
    plt.show()

def plot_r2_hist(df, xlim=None, bins=100, tickfontsize=15, labelfontsize=15, figsize=(10, 5)):
    """R2のDOSを図示する．
    plt.show()を実行しない。

    Args:
        df (pd.DataFrame): データ
        xlim (tuple(float, float), optional): 図のx range. Defaults to None.
        bins (int, optional): histogramのbin数. Defaults to 100.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
        figsize (tuple, optional): figure size. Defaults to (10,5).
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = df["score_mean"]
    occurrence, edges = np.histogram(x, bins=bins, range=xlim)
    left = (edges[:-1]+edges[1:])*0.5
    ax.bar(left, occurrence, width=left[1]-left[0])
    # df.hist("score_mean", bins=100, xlim=xlim, ax=ax)
    ax.set_xlabel("$R^2$", fontsize=labelfontsize)
    ax.set_ylabel("occurrence", fontsize=labelfontsize)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    fig.tight_layout()
    return fig, ax


def calculate_coeffix(descriptor, combilist, coeflist):
    """表示のために 係数０の部分を加えて係数を作りなおす．

    Args:
        descriptor (list[str]): all the descriptor names
        combilist (list): a list of descriptor combinations of the models
        coeflist (np.array): a list of coefficients of the models

    Returns:
        list: a list of coefficnets whose length is the same as the length of all the descriptors
    """
    n = len(descriptor)
    coeffixlist = []
    for combi, coef in zip(combilist, coeflist):

        coeffix = np.zeros((n))
        # if combi=[1,2], and coef=[val1,val2], then coeffix=[0,val1,val2,0,0]
        for i, id in enumerate(combi):
            coeffix[id] = coef[i]

        # 都合でlistに直す．
        coeffixlist.append(list(coeffix))
    return coeffixlist


def plot_weight_diagram(df_result, descriptor_names, ax=None, nmax=200, figsize=(10, 5)):
    """weight diagramの表示とデータフレーム出力を行う。

    Args:
        df_result (pd.DataFrame): dataimport pandas as pd
        descriptor_names (List[str]): 説明変数カラムリスト
        ax (matplotlib.axes): axes. Defaults to None.
        nmax (int, optional): the maximum number of the data to show. Defaults to 200.
        figsize (tuple, optional): figure size. Defaults to (10,5).
    """
    ax_orig = ax
    x = df_result.loc[:nmax, descriptor_names].values

    x = np.log10(np.abs(x))
    df_x = pd.DataFrame(x, columns=descriptor_names).replace(
        [-np.inf, np.inf], np.nan)
    df_weight_diagram = df_x.fillna(-3)
    if ax_orig is None:
        fig, ax = plt.subplots(figsize=figsize)
    # ax.set_title("log10(abs(coef))")
    sns.heatmap(df_weight_diagram.T, ax=ax)
    ax.set_ylim((-0.5, df_weight_diagram.shape[1]+0.5))
    if ax_orig is None:
        fig.tight_layout()
    plt.show()

def make_counts(df_result, descriptor_names, sentense, ratio=False):
    """
    説明変数が用いられた回数を計算する．

    Args:
        df_result (pd.DataFrame): データ
        descriptor_names (list[str]): 説明変数名リスト．
        sentense (str): query文
        ratio (bool, optional): 回数(False), 割合(True)を返す． Defaults to False.

    Returns:
        pd.DataFrame: 回数もしくは割合データ．
    """
    x = df_result[descriptor_names].values != 0  # 係数が０でない．＝その説明変数が含まれるモデル．
    df_indicator_diagram = df_result.copy()
    df_indicator_diagram.loc[:, descriptor_names] = x

    dfq = df_indicator_diagram.query(sentense)
    print(sentense, "#=", dfq.shape[0])
    if ratio:
        return np.sum(dfq[descriptor_names], axis=0)/dfq.shape[0]
    else:
        return np.sum(dfq[descriptor_names], axis=0)


def make_and_plot_block_weight_list(df_result, descriptor_names, querylist, figsize=(10, 5)):
    """
    querylistのblock weight diagramを計算する．

    Args:
        df_result (pd.DataFrame): データ．
        descriptor_names (list[str]): 説明名リスト．
        querylist (list[str]): query文リスト．
        figsize (tuple, optional): figure size. Defaults to (10,5).
    Returns:
        pd.DataFrame: block weight diagram.
    """
    result = []
    for sentense in querylist:
        # 前の図に合わせるためにdescriptor_namesの順序を逆にする．
        t = make_counts(
            df_result, descriptor_names[::-1], sentense, ratio=True)
        result.append(t)
    dfq = pd.DataFrame(result, index=querylist)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dfq.T, ax=ax)  # 前の図に合わせるためにtransposeする．
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
    fig.tight_layout()
    plt.show()

    return dfq


def make_indicator_diagram(df_result, descriptor_names, nmax=50):
    """indicator diagramを作成する．

    Args:
        df_result (pd.DataFrame): data
        nmax (int, optional): the maximum number of the data to show. Defaults to 50.
    """
    x = df_result[descriptor_names].values != 0
    df_indicator_diagram = pd.DataFrame(x, columns=descriptor_names)
    return df_indicator_diagram


def make_df_by_index(df_indicator_diagram, descriptor_names, index):
    """部分データを得るためのデータを作る．

    return valueはindexで指定された範囲で，
    descriptor_names（０でない数）+"N"（データインスタンス総数）をカラムに持つデータになる．

    Args:
        df_indicator_diagram (pd.DataFrame): indicator diagram データ
        descriptor_names (list[str]): 説明変数名リスト
        index (list[int]): データインスタンスの部分indexリスト

    Returns:
        pd.DataFrame: indicator diagram部分データ
    """
    dfq = df_indicator_diagram.iloc[index, :]
    print("#=", dfq[descriptor_names].shape[0])
    df_all = pd.DataFrame({"N": [dfq[descriptor_names].shape[0]]},)
    dfq_sum = dfq[descriptor_names].astype(int).sum(axis=0)
    df1 = pd.DataFrame(dfq_sum).T

    return pd.concat([df1, df_all], axis=1)
    # print(np.sum(dfq[descriptor_names], axis=0))


def make_all_ind_by_index(df_indicator_diagram, descriptor_names, regionindex, regionsize):
    """各領域の非ゼロの説明変数の割合を得る．

    regionindex=[0,1,..,N]
    for i in regionindex:
        region = [ i*regionsize, (i+1)*regionsize ]
    と各data instance index領域を定義する．

    Args:
        df_indicator_diagram (pd.DataFrame): データ
        descriptor_names (list[str]): 説明変数名リスト
        regionindex (list[int])): 領域インデックスリスト
        regionsize (int)): 領域サイズ

    Returns:
        pd.DataFrame: 分割領域ごとのデータ
    """
    df_ind_list = []
    for i in regionindex:
        region = list(range(i*regionsize, (i+1)*regionsize))
        df_ind = make_df_by_index(
            df_indicator_diagram, descriptor_names, region)
        df_ind_list.append(df_ind)
    _df = pd.concat(df_ind_list, axis=0).reset_index(drop=True)

    names = list(_df.columns)
    names.remove("N")
    v0 = _df["N"]
    for name in names:
        _df[name] = _df[name]/v0

    if False:
        fig, ax = plt.subplots()
        _df[names].T.plot(ax=ax)
        ax.set_ylabel("frequency")
        ax.set_xticks(list(range(len(names))))
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim((0, 1))
    return _df


def plot_df_imp_by_index(df_imp_by_index, descriptor_names, regions, regionsize,
                         tickfontsize=15, legendfontsize=12, labelfontsize=15, markersize=10, figsize=(10, 5)):
    """各領域の説明変数の頻度を図示する．

    Args:
        df_imp_by_index (list[pd.DataFrame]): _description_
        descriptor_names (list[str]): 説明変数名リスト
        regions (list[int])): 領域リスト
        regionsize (int): 領域サイズ
        comment (str): 表示用コメント. Defaults to "importancebyindex".
        metadata (dict): 表示用データ. Defaults to G_METADATA. 
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): ticks font size. Defaults to 15.
        markersize (int, optional): marker size. Defaults to 10.
        figsize (tuple, optional): figure size. Defaults to (10, 5).
    """
    xticks_str = []
    for i in regions:
        xticks_str.append("[{}:{})".format(i*regionsize, (i+1)*regionsize))
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if False:
        df_imp_by_index[descriptor_names].plot(marker="o", ax=ax)
    else:
        marker_list = [".", "o", "v", "^", "<", ">"]
        marker_list += ["8", "s", "p", "*", "h", "H", "+", "x", "D", "d"]
        for exp_name, marker in zip(descriptor_names, marker_list):
            df_imp_by_index[exp_name].plot(marker=marker, markersize=markersize, ax=ax)
    ax.set_xticks(list(range(len(regions))))
    ax.set_xticklabels(xticks_str, rotation=90)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legendfontsize)
    ax.set_ylabel("occurrence", fontsize=labelfontsize)
    fig.tight_layout()
    plt.show()


def plot_weight_diagram3d(df, descriptor_names, index=(0, 100), elev=30, azim=160, rotation=-90, figsize=(10, 10)):
    """plot weght diagram 3D in linear z scale.

    Args:
        df (pd.DataFrame): data.
        descriptor_names (list, optional): df.iloc[index,descriptor_names].
        index (tuple, optional): select df.iloc[index,descriptor_names]. Defaults to (0, 100).
        elev (int, optional): elev of view_init(elev=elev, azim=azim). Defaults to 30.
        azim (int, optional): azim of view_init(elev=elev, azim=azim). Defaults to 160.
        rotation (float, optional): yticklabels rotation value. Defaults to -60.
        figsize (tuple, optional): figure size. Defaults to (10, 10).
    """
    _df = df.loc[index[0]:index[1], descriptor_names].copy()
    def f_abs(x): return np.abs(x)
    _df = _df.apply(f_abs)
    sum_value = _df.T.sum().values
    _df = _df/sum_value.reshape(-1, 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, name in enumerate(descriptor_names):
        z = _df[name].values
        y = np.full(fill_value=i, shape=z.shape)
        x = np.array(_df.index)
        verts = [(x[i], y[i], z[i]) for i in range(len(x))] + [(x.max(), i, 0), (x.min(), i, 0)]
        ax.add_collection3d(Poly3DCollection([verts], color='orange', alpha=0.1))
        ax.plot(x, y, z)

    ax.set_xlabel("index")
    ax.set_ylabel("descriptor")
    ax.set_zlabel("importance")
    ax.set_yticks(np.arange(len(descriptor_names)))
    ax.set_yticklabels(descriptor_names, rotation=rotation)
    ax.view_init(elev=elev, azim=azim)
    plt.show()

import os

def make_combination_stacked_bar(df, target="score_mean", column_names=("C_R", "C_T", "vol_per_atom"),
                                 action=("occurrence", "fraction"), nbins=50, figsize=(10, 5), 
                                 output_dir=None, img_format="png"):
    """make stacked bar plot and fraction stacked bar plot.

    It deletes combinations whose size is 0.

    Args:
        df (pd.DataFrame): data contaning R2 of descriptor combinations.
        target (str, optional): score name. Defaults to "score_mean".
        column_names (tuple, optional): column names to make combinations. Defaults to ("C_R", "C_T", "vol_per_atom").
        action (tuple, optional): kind of plots. Defaults to ("occurrence", "fraction").
        nbins (int, optional): number of bins of histogram. Defaults to 50.
        figsize (tuple, optional): figure size. Defaults to (10,5).
        output_dir (str, optional): output directory. Defaults to None.
        img_fomat (str, optional): image file format. Defaults to "png".
    """
    _df = df.copy()
    r2_lim = (_df[target].min(), _df[target].max())

    df_merged = None
    column_names = set(column_names)
    for icombi in range(1, len(column_names)+1):
        for use_set in itertools.combinations(column_names, icombi):
            complement = column_names - set(use_set)
            sentense = []
            for x in use_set:
                sentense.append(f"{x}!=0")
            for x in complement:
                sentense.append(f"{x}==0")
            s = " and ".join(sentense)
            __df = _df.query(s)
            if __df.size == 0:
                continue
            occur, edge = np.histogram(__df[target].values, bins=nbins, range=r2_lim)
            x = (edge[:-1]+edge[1:])*0.5
            df_toadd = pd.DataFrame({target: x, s: occur})
            if df_merged is None:
                df_merged = df_toadd
            else:
                df_merged = pd.merge(df_merged, df_toadd, on=target)  # , how="left") #r2をキーとしてaxis=1方向に結合する。

    r2_values = df_merged[target].apply(lambda x: "%.3f" % (x))
    df_merged[target] = r2_values
    df_bar = df_merged.set_index(target, drop=True)
    if "occurrence" in action:
        fig, ax = plt.subplots()
        df_bar.plot(kind="bar", stacked=True, figsize=figsize, ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("occurrence")
        fig.tight_layout()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = os.path.join(output_dir, f"combination_stacked_bar_occurrence.{img_format}")
            fig.savefig(img_filename)
            print(img_filename,"is made.")
        plt.show()

    sum_values = df_bar.T.apply(lambda x: np.sum(x)).values
    df_stacked_bar = df_bar / sum_values.reshape(-1, 1)
    if "fraction" in action:
        fig, ax = plt.subplots()
        df_stacked_bar.plot(kind="bar", stacked=True, figsize=figsize, ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("fraction")
        fig.tight_layout()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            img_filename = os.path.join(output_dir, f"combination_stacked_bar_fraction.{img_format}")
            fig.savefig(img_filename)
            print(img_filename,"is made.")
        plt.show()
