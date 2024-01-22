
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd

def plot_atomicprop(df, df_select, xlabel, ylabel, materialkey="index",
        tickfontsize=15, labelfontsize=14, textfontsize=15,):
    """plot atom properties.

    Args:
        df (pd.DataFrame): データ.
        df_select (pd.DataFrame): 選択されたデータ.
        xlabel (str): x label column name.
        ylabel (str): y label column name.
        materialkey (str): "element" column name. Defaults to "index".
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
        textfontsize (int, optional): text font size. Defaults to 15.
    """
    fig, axes = plt.subplots(1, 2, dpi=300)
    for i, ax in enumerate(axes):
        if i == 0:
            _x = df[xlabel].values
            _y = df[ylabel].values
            _m = df[materialkey].values
        else:
            _x = df_select[xlabel].values
            _y = df_select[ylabel].values
            _m = df_select[materialkey].values
        ax.scatter(_x, _y)
        for __x, __y, __m in zip(_x, _y, _m):
            ax.text(__x, __y, __m, fontsize=textfontsize)
        ax.set_xlabel(xlabel, fontsize=labelfontsize)
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        ax.tick_params(axis = 'x', labelsize =tickfontsize)
        ax.tick_params(axis = 'y', labelsize =tickfontsize)   
    fig.tight_layout()