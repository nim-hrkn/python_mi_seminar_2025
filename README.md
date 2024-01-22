
# CONTENT

このレポジトリには
２０２３年度マテリアルズ・インフォマティクス連続セミナー 第二版
のJupyter notebook/Lab スクリプトおよびデータが含まれます．

Python/scikit-learnを用いた機械学習手法の紹介を行います。

参考：
「Pythonではじめるマテリアルズインフォマティクス」
木野 日織，ダム ヒョウ チ（著）近代科学社Digital


## スライド及び動画

スライド及び動画は以下のurlにあります。

２０２３年度マテリアルズ・インフォマティクス連続セミナーを一部改定して作成しています。そのため一部スライドの年号や回数が修正されておりません。また、呼び方が「連続セミナー」、「チュートリアル」と混乱しています。ご了承ください。

### 基礎編

- 座学：https://www.docswell.com/s/3465680103/ZWPN45-2022-12-07-091431, https://youtu.be/WkStbYrGCM4

- （追加）最低限のPython package紹介,データ紹介,LLMによる知識獲得：https://www.docswell.com/s/3465680103/ZJLP43-2024-01-22-142423, https://youtu.be/Oz8hED87qiQ
- （改定）回帰、交差検定、LLMによるソースコード作成：https://www.docswell.com/s/3465680103/Z4Q1VD-2024-01-22-142634, https://youtu.be/3e698MCBlng
- （改定）次元圧縮、分類、クラスタリング、LLMによるソースコード作成：https://www.docswell.com/s/3465680103/5M1P7W-2024-01-22-142812, https://youtu.be/Poubb2yzdZU

### 応用編

基礎編を終了してから視聴することを想定しています。
基礎編を修正しているので一部の説明が欠けている場合があります。

- 次元圧縮を併用したクラスタリング、トモグラフ像の復元
https://www.docswell.com/s/3465680103/K3R1V5-2023-01-09-214658, https://youtu.be/4CEa3mb1vug
- 説明変数重要性、全探索を用いた説明変数重要性：https://www.docswell.com/s/3465680103/ZMJJ9Z-2023-01-16-123943, https://youtu.be/afg_2sIG3O8
- ベイズ最適化、推薦システム：https://www.docswell.com/s/3465680103/5YY1D5-2023-01-23-142130, https://youtu.be/bE-kfA_Z3z0

## 補足資料
- bitbucket repositoryのダウンロードの仕方： https://youtu.be/xkgHmyenjC4 .
- Jupyter labの使い方： https://youtu.be/WIw_xR6zFjs .
- クラスタリング妥当性：https://www.docswell.com/s/3465680103/ZXLG2Z-2022-12-22-140805 .
- ベイズ最適化ツール：https://www.docswell.com/s/3465680103/587MQK-2022-12-20-120642 .
- 第五回の質問への解答: https://www.docswell.com/s/3465680103/Z1P1GK-2023-01-18-152055 .
- 日本のAI戦略：セミナーの位置づけ、他の勉強ソース： https://www.docswell.com/s/3465680103/ZNW8DZ-2022-12-07-172019 .

# UPDATES due to the obsolte functions in scikit-learn

## Dec. 21, 2022
- sns.kdeplot(array1, array2, ...) -> sns.kdeplot(x=array1, y=array2, ...)

# LICENSE

Copyright (c) 2022-2023 Hiori Kino<br>
Released under the MIT license<br>
https://opensource.org/licenses/mit-license.php
