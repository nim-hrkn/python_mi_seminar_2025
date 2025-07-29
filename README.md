
# マテリアルズ・インフォマティクス連続ハンズオン（2025年度）

このリポジトリには、2025年度マテリアルズ・インフォマティクス連続ハンズオンにおける Jupyter Notebook / Lab スクリプト および 関連データ を収録しています。Python / scikit-learn / PyTorch を用いた機械学習手法の基礎から応用までを扱います。


## 

本スクリプトは以下の環境で動作確認を行っています：

- Python 3.10.17

- scikit-learn 1.6.1

- SciPy 1.15.3

- PyTorch 2.7.0

※パッケージのバージョン更新等により、全ての環境での動作保証はできかねます。予めご了承ください。

##  対象

### 想定受講者

Python の基本文法を理解しており、これから scikit-learn や PyTorch を用いて機械学習を実践したい方

大規模言語モデル（LLM） を活用して Python による可視化や機械学習コードを効率的に書きたい方

※受講の判断には、「スライドおよび動画」セクションをご参照ください。


### 想定しない対象者

- GUIベースのソフトウェアでの操作を希望される方 → 代替として Orange Data Mining を用いたチュートリアル をご参照ください。

## スライド及び動画

スライド及び動画は以下のurlにあります。

２０２２年度ー２０２４年度マテリアルズ・インフォマティクス連続セミナーを一部改定して作成しています。
そのため一部スライドの年号や回数が修正されておりません。
また、呼び方が「連続セミナー」、「チュートリアル」,「ハンズオン」と混乱しています。ご了承ください。

### 座学編

目的：機械学習手法の基礎を簡単に知る。

- 座学：スライド https://www.docswell.com/s/3465680103/ZWPN45-2022-12-07-091431, 動画 https://youtu.be/WkStbYrGCM4

座学は下で紹介している「Orange Dataminingを用いた機械学習手法チュートリアル」の一部も参照できます。

### 基礎編

目的１：Pythonを用いたscikit-learnライブラリの機械学習手法の基礎的な使い方を知る。<br>
目的２：LLM を用いたコード支援活用法を学ぶ。

- 最低限のPython package紹介,データ紹介,LLMによる知識獲得：スライド https://www.docswell.com/s/3465680103/ZJLP43-2024-01-22-142423, 動画 https://youtu.be/Oz8hED87qiQ
- 回帰、交差検定、LLMによるソースコード作成：スライド https://www.docswell.com/s/3465680103/Z4Q1VD-2024-01-22-142634, 動画 https://youtu.be/PgV0ZMqJWTI
- 次元圧縮、分類、クラスタリング、LLMによるソースコード作成：スライド https://www.docswell.com/s/3465680103/5M1P7W-2024-01-22-142812, 動画 https://youtu.be/Poubb2yzdZU

### 応用編

基礎編を終了してから視聴することを想定しています。
一部説明が簡略化されています。

目的：基礎編の知識を元に機械学習手法のより高度な使い方を知る。

- 次元圧縮を併用したクラスタリング、トモグラフ像の復元：
スライド https://www.docswell.com/s/3465680103/K3R1V5-2023-01-09-214658, 動画 https://youtu.be/4CEa3mb1vug
- 説明変数重要性、全探索を用いた説明変数重要性：スライド https://www.docswell.com/s/3465680103/ZMJJ9Z-2023-01-16-123943, 動画　https://youtu.be/afg_2sIG3O8
- ベイズ最適化、推薦システム：スライド https://www.docswell.com/s/3465680103/5YY1D5-2023-01-23-142130, 動画 https://youtu.be/bE-kfA_Z3z0


- (追加、修正)LLMによるニューラルネットワークモデル手法の学習：スライド （正）https://mat-dacs.dxmt.mext.go.jp/a31, （副）https://www.docswell.com/s/3465680103/5R61YR-2025-07-28-213446, 動画 https://youtu.be/vldEtevFBfk


## 実行環境の構築と実行

### 実行環境の構築

各自のPCでスクリプトを動作Python3環境下でのjupyter lab, scikit-learnなどのパッケージのインストールをお願いします。

#### Anaconda Distributionを用いる場合
Anaconda は、Python本体および科学技術計算に必要な基本的パッケージ（NumPy, pandas, scikit-learn など）を一括で導入できる便利なディストリビューションです。

https://www.anaconda.com/download​

Anaconda DistributionのFree Plan適用条件に関してはこちらに記載があります。（組織によっては無料で試用できません。）
https://legal.anaconda.com/policies/en/?name=terms-of-service

ただし、Anaconda には PyTorch や progressbar などの一部パッケージは含まれていません。そのため、インストール後にこれらを追加する必要があります。

#### 他のpackageを自分で構築する場合
Anaconda より軽量な構成を望む場合は、以下の選択肢もあります：

- miniconda:
- miniforge: conda-forge に特化した軽量なディストリビューション
- 各自仮想環境を構築

どちらも仮想環境を活用して構築することを推奨します。これらを用いた場合も、Jupyter Lab、scikit-learn、PyTorch、progressbar などを個別にインストールする必要があります。


### jupyter Labの使い方

jupyter Labの簡単な使い方の説明はこちらにあります。
https://youtu.be/WIw_xR6zFjs

## Orange Data Miningを用いた機械学習手法チュートリアル

Pythonを用いないワークフローによる機械学習手法適用ソフトOrange Data Miningを用いた機械学習手法チュートリアル

https://bitbucket.org/kino_h/orange_mi_seminar_2023/src/main/


## 補足資料
- bitbucket repositoryのダウンロードの仕方： https://youtu.be/xkgHmyenjC4 .
- Jupyter labの使い方： https://youtu.be/WIw_xR6zFjs .
- クラスタリング妥当性：https://www.docswell.com/s/3465680103/ZXLG2Z-2022-12-22-140805 .
- ベイズ最適化ツール：https://www.docswell.com/s/3465680103/587MQK-2022-12-20-120642 .
- 第五回の質問への解答: https://www.docswell.com/s/3465680103/Z1P1GK-2023-01-18-152055 .
- 日本のAI戦略：セミナーの位置づけ、他の勉強ソース： https://www.docswell.com/s/3465680103/ZNW8DZ-2022-12-07-172019 .

# UPDATES 

## Jul. 29, 2025
mean_squared_error(y_true, y_pred, squared=False)
にてsquared=を使わないように修正。

## Feb. 25, 2025
- 200.NN/の追加
- plt.show()とする。

## Dec. 21, 2022
- sns.kdeplot(array1, array2, ...) -> sns.kdeplot(x=array1, y=array2, ...)、due to the obsolte functions in scikit-learn

## Jan. 22, 2023
- 基礎編の３つを追加、改定。

# 参考書籍

- 「改訂版 Pythonではじめるマテリアルズインフォマティクス ChatGPTを活用しよう」
木野 日織，ダム ヒョウ チ（著）[近代科学社Digital、ISBN:9784764960466](https://www.kindaikagaku.co.jp/book_list/detail/9784764961005/)

- 「Orange Data Mining ではじめるマテリアルズインフォマティクス」⽊野 ⽇織、ダム ヒョウ チ(著) [近代科学社、ISBN:9784764906310](https://www.kindaikagaku.co.jp/book_list/detail/9784764906310/)


# 免責事項
本セミナーのスクリプトやデータを用いて得られた結果について一切の責任を持ちません。

# LICENSE

Copyright (c) 2022-2025 Hiori Kino<br>
Released under the MIT license<br>
https://opensource.org/licenses/mit-license.php
