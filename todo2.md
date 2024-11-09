

B.2 Supervised Finetuning

私たちはCrossDocked バージョン1.3を利用し, 各分子について, GNINAモデルによって最適化されたものを除くminimizedとdockedのファイルについて, 全ての中間の座標を取り出した。
    ... dockedと.gninatypesって同じものだと思っていたが, 違うのか?
それぞれの分子について, Prodyツールを使ってポケットを抽出した。その結果, 2700万のポケット-リガンドペアが得られた。

Finetuningにおいては, いくつかのハイパラを除き事前学習と同じ学習手法を用いた。違いとしては, 最大lrを5x10^-4とし, warmupは100stepとした。
SMILESの重みを1, 座標の重みを5とし, コンテキストであるポケットの重みは0とした。
3.3.1で述べた通り, SMILESのランダム化を行い, ポケットを回転させた。また, リガンドの座標の中心が原点になるようにした。


[現行と違うところ]
- minimized, dockedファイルの全ての配座を使っている。
- それぞれについて, Prodyツールを使ってポケットを抽出している。
    - ... 分子ごとなのか, 配座ごとなのか?
1- SMILESの重みが1, 座標の重みが5, コンテキストの重みが0
1- SMILESのランダム化
1- ポケットをランダムに回転
1- リガンドの座標の中心を原点にする。

- 最大lr=5e-4,  warmup=100

We use the public CrossDocked version v1.3 and for each molecule (except the ones optimized by the Gnina model (McNutt et al., 2021) as it yields too many bad itermediate samples) we take its “minimized” and “docked” formats and extract all intermediate molecules from their files. For each such molecule we cut the pocket with the ProDy (Bakan et al., 2011) tool. As a result of this process we obtain around 27M pocket-ligand pairs. The size of the CrossDocked that we use is around 50GB.
We use the same recipe for finetuning as for the pretraining with a few changes in hyper- parameters. In particular, we use maximal learning rate of 5 × 10−4 and only 100 warump
steps. The learning rate schedule, weight decay, optimizer, maximal gradient norm are the same as for the pretraining. The only substantial difference from the pretraining stage is the weighted loss which we use for the CrossDocked finetuning. Specifically, we weight tokens that correspond to different parts of the output, differently. For example, the SMILES tokens have the weight of 1 while tokens that correspond to the XYZ coordinates placed after SMILES have the weight of 5. The tokens corresponding to the pocket have the weight of 0 since they are used as the context only and we don’t intend to generate them. As it was described in the Section 3.3.1, we do SMILES randomization (see Bjerrum (2017) for implementation details) and rotate pocket it’s ligand randomly - first we sample a random 3D rotation vector, we convert it to a rotation matrix and apply it to the coordinates of both. Also, we enforce the origin of their coordinates to be the same, namely, the coordinate center of the ligand (i.e. we guarantee that the model will generate coordinates around the origin). We train the model on the CrossDocked dataset for 1 epoch. As it was mentioned in Section 3.3.1, we extract the full version of the CrossDocked data.
For the finetuning on the GEOM-DRUGS dataset, we use the same hyperparameters as in the SFT stage for CrossDocked with only two differences. First, we weight the loss for all tokens with the same weight of 1. Second, we don’t rotate 3D coordinates of the molecule but only do SMILES randomization.