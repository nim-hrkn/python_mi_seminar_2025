
# CONTENT

このレポジトリには
２０２４年度マテリアルズ・インフォマティクス連続ハンズオン
のJupyter notebook/Lab スクリプトおよびデータが含まれます。
Python/scikit-learnを用いた機械学習手法の紹介を行います。

本スクリプトはPython 11.8, scikt-learn 1.4.1.post1, 
scipy 1.9.3, pytorch 2.4.0で動作確認をしています。
パッケージの仕様は変わっていくので、全ての場合に動作確認はできません。
ご了承ください。

##  対象

### 想定受講者

- Pythonは知っているが、これからPython/scikit-learnを用いて機械学習手法を適用したい方。
- 大規模言語モデルを用いてscikit-learnや可視化のPythonコードの書き方を知りたい方。

「スライド及び動画」節で示しているスライドを見て受講を決めてください。


### 想定外


- Pythonを使わずにGUIソフトを使いたい人 → Orange Data Miningを用いた連続セミナー（下で紹介）。

## スライド及び動画

スライド及び動画は以下のurlにあります。

２０２２年度、２０２３年度マテリアルズ・インフォマティクス連続セミナーを一部改定して作成しています。
そのため一部スライドの年号や回数が修正されておりません。
また、呼び方が「連続セミナー」、「チュートリアル」,
「ハンズオン」と混乱しています。ご了承ください。

### 座学編

目的：機械学習手法の基礎を簡単に知る。

- 座学：https://www.docswell.com/s/3465680103/ZWPN45-2022-12-07-091431, https://youtu.be/WkStbYrGCM4

座学は下で紹介している「Orange Dataminingを用いた機械学習手法チュートリアル」の一部も参照できます。

### 基礎編

目的１：Pythonを用いたscikit-learnライブラリの機械学習手法の基礎的な使い方を知る。<br>
目的２：大scikit-learnライブラリの使用支援・コード作成支援のための規模言語モデルの使い方を知る。

- 最低限のPython package紹介,データ紹介,LLMによる知識獲得：https://www.docswell.com/s/3465680103/ZJLP43-2024-01-22-142423, https://youtu.be/Oz8hED87qiQ
- 回帰、交差検定、LLMによるソースコード作成：https://www.docswell.com/s/3465680103/Z4Q1VD-2024-01-22-142634, https://youtu.be/PgV0ZMqJWTI
- 次元圧縮、分類、クラスタリング、LLMによるソースコード作成：https://www.docswell.com/s/3465680103/5M1P7W-2024-01-22-142812, https://youtu.be/Poubb2yzdZU

### 応用編

基礎編を終了してから視聴することを想定しています。
基礎編を修正しているので一部の説明が欠けている場合があります。

目的：基礎編の知識を元に機械学習手法のより高度な使い方を知る。


- 次元圧縮を併用したクラスタリング、トモグラフ像の復元
https://www.docswell.com/s/3465680103/K3R1V5-2023-01-09-214658, https://youtu.be/4CEa3mb1vug
- 説明変数重要性、全探索を用いた説明変数重要性：https://www.docswell.com/s/3465680103/ZMJJ9Z-2023-01-16-123943, https://youtu.be/afg_2sIG3O8
- ベイズ最適化、推薦システム：https://www.docswell.com/s/3465680103/5YY1D5-2023-01-23-142130, https://youtu.be/bE-kfA_Z3z0
- (追加)LLMによるニューラルネットワークモデルの基礎：https://www.docswell.com/s/3465680103/KP2XYG-2025-02-25-203000, https://youtu.be/Ln5y8Hjzexg

## 実行環境の構築と実行

### 実行環境の構築

各自のPCでスクリプトを動作Python3環境下でのjupyter lab, scikit-learnなどのパッケージのインストールをお願いします。

例えばAnaconda Distributionにより使用するPython環境一式をインストールできます。​
https://www.anaconda.com/download​

Anaconda DistributionのFree Plan適用条件に関してはこちらに記載があります。
https://legal.anaconda.com/policies/en/?name=terms-of-service

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

## Feb. 25, 2025
- 200.NN/の追加
- plt.show()とする。

## Dec. 21, 2022
- sns.kdeplot(array1, array2, ...) -> sns.kdeplot(x=array1, y=array2, ...)、due to the obsolte functions in scikit-learn

## Jan. 22, 2023
- 基礎編の３つを追加、改定。

# 参考書籍

- 「Pythonではじめるマテリアルズインフォマティクス」
木野 日織，ダム ヒョウ チ（著）近代科学社Digital、ISBN:9784764960466

- 「Orange Data Mining ではじめるマテリアルズインフォマティクス」⽊野 ⽇織、ダム ヒョウ チ(著) 近代科学社、ISBN:9784764906310


# 免責事項
本セミナーのスクリプトやデータを用いて得られた結果について一切の責任を持ちません。

# LICENSE

Copyright (c) 2022-2023 Hiori Kino<br>
Released under the MIT license<br>
https://opensource.org/licenses/mit-license.php
