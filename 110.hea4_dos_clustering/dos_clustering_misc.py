import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def plot_each_DOS(df, logdos_names,  target_name, figsize=(8,5), n1=4, n2=3, filename=None):
    """plot DOS separated by target_name
    
    Args:
        df (pd.DataFrame): data.
        logdos_names (str): 説明変数.
        target_name (str): 目的変数.
        figsize (Tuple(float, float)): figure size. Defaults to (8.5).
        n1 (int, optional): figure panel rows. Defaults to 4.
        n2 (int, optional): figure panel columns. Defaults to 3.
        filename (str): image filename. Defaults to None.
    """
    y_uniq = np.unique(df[target_name].values)
    n1, n2 = 4, 3
    fig, axes = plt.subplots(n1, n2, figsize=(8,5))
    for i in range(n1):
        for j in range(n2):
            yval = y_uniq[i*n2+j]
            _df = df[df[target_name]==yval]
            _X = _df[logdos_names]
            ax = axes[i,j]
            ax.plot(_X.T, c="black", alpha=0.01)
            ax.set_title(yval)
            # ax.tick_params(left = False, bottom = False)
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off        
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

def plot_X2_compare(X2, y, yp, figsize=(6,3), alpha=0.3, fontsize=10,
                    filename: str=None):
    """plot (X2,y) and (X2,yp).

    Args:
        X2 (np.ndarray): size (N,P), explanatory variables for visualization.
        y (np.ndarray):  size (N), target varaibles.
        yp (np.ndaray): size (N), predicted variables.
        figsize (List[float], optional): figure size. Defaults to (10,5).
        marker (str, optional): marker symbol. Defauls to ".".
        alpha (float, optional): alpha value. Defaults to 0.3.
        fontsize (float, optional): fontsize. Defaults to 10.   
        filename (str, optional): filename. Defaults to None.
    """

    marker_list = ["o", "v", "^", "<", ">"]
    marker_list += ["8", "s", "p", "*", "h", "H", "+", "x", "D","d"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    y_uniq = np.unique(y)
    for yval,marker in zip(y_uniq, marker_list):
        ilist = y == yval
        ax.plot(X2[ilist, 0], X2[ilist, 1], marker, label=yval, alpha=alpha)
        center = (X2[ilist, 0].mean(), X2[ilist, 1].mean())
        ax.text(center[0], center[1], yval, fontsize=fontsize)
    ax = axes[1]
    yp_uniq = np.unique(yp)
    for ypval, marker in zip(yp_uniq, marker_list):
        ilist = yp == ypval
        ax.plot(X2[ilist, 0], X2[ilist, 1], marker, label=ypval, alpha=alpha)
        center = (X2[ilist, 0].mean(), X2[ilist, 1].mean())
        ax.text(center[0], center[1], ypval, fontsize=fontsize)
    if filename is not None:
        print(filename)
        fig.savefig(filename)

def assign_frequent_value(y, yp):
    """assign the most frequent value to yp

    Args:
        y (np.ndarray): y.
        yp (np.ndarray): predicted y.

    Returns:
        nd.ndarray: reassigned yp.
    """
    yp_uniq = np.unique(yp)
    convertdic = {}
    for ypval in yp_uniq:
        ilist = yp == ypval
        y1 = y[ilist]
        y1freq = Counter(y1)
        key = y1freq.most_common()[0][0]
        convertdic[ypval] = key
    print(convertdic)
    yp_freq = []
    for yp1 in yp:
        a = convertdic[yp1]
        yp_freq.append(a)

    return np.array(yp_freq)


def smear_dos(v, newn, alpha, type_=1):
    """smear counter by 

    exp(-alpha((r-i)^2)) if type_==1,
    exp(-alpha((r-i)^2))*(r-i)^2 if type_==2,
    tanh(alpha((r-i)) if type_==3,

    Args:
        v (np.ndarray): values
        newn (int): new number of divisions
        alpha (float, optoinal): exp(-alpha*i*2). Defaults 1.0.

    Returns:
        np.ndarrays: smeared values
    """
    n = v.shape[1]

    i = [i for i in range(n)]
    i = np.array(i)/n

    r = np.linspace(0, 1, newn)

    r_i = r[np.newaxis, :]-i[:, np.newaxis]

    if type_ == 1:
        expr_r2 = np.exp(-alpha*r_i**2)
    elif type_ == 2:
        expr_r2 = np.exp(-alpha*r_i**2)*r_i**2
    elif type_ == 3:
        expr_r2 = np.tanh(alpha*r_i)
    else:
        raise ValueError("uknown type_={}".format(type_))

    expr_r2 = expr_r2[np.newaxis, :, :]

    v = v[:, :, np.newaxis]

    expr_r2_v = expr_r2 * v
    v2 = expr_r2_v.sum(axis=1)

    return v2, expr_r2


def add_convolution_variables(df, descriptor_names, n_new, alpha=None, metadata=None,
    tickfontsize=15):
    """add convolution variables

    if metadata is given, the png image will be written.

    Args:
        df_obs (pd.DataFrame): データ．
        descriptor_names: 説明変数名リスト．
        n_new (int): 説明変数数．
        metadata (dict, optional): 表示用データ. Defaults to None.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        
    Returns:
        tuple containinig

        - pd.DataFrame: データ
        - list[str]: descriptor_names
    """
    df = df.copy().reset_index(drop=True)
    X = df[descriptor_names].values

    if alpha is None:
        alpha = int(50*((n_new/5)**2))
    v2, expr_r2 = smear_dos(df[descriptor_names].values, n_new, alpha=alpha)

    fig, ax = plt.subplots()
    ax.plot(expr_r2[0, :, :])
    ax.tick_params(axis = 'x', labelsize =tickfontsize)
    ax.tick_params(axis = 'y', labelsize =tickfontsize)    
    fig.tight_layout()
    # plt.title("expr_r2")
    if metadata is not None:
        filename = "_".join([metadata["prefix"], metadata["dataname"], metadata["df_type"],
                            str(metadata["n_smeared_dos"]), "smeardos"])+".png"
        print(filename)
        fig.savefig(os.path.join(metadata["outputdir"], filename))

    smeared_names = []
    for i in range(n_new):
        smeared_names.append("smeared_dos{}".format(i))

    df_smeared = pd.DataFrame(v2, columns=smeared_names)
    return pd.concat([df, df_smeared], axis=1), smeared_names
