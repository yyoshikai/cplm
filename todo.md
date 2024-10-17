
<機能的に必要なこと>
座標の範囲がどのくらいかを調べる。
    -200~200くらいまであった。

<速度など余裕があれば>
・positional embeddingをクラス変数にする
    state_dictで読み込まないようになどしたい。

We employ Flash-Attention2 Dao (2023) and DeepSpeed optimization accelerator. 
As the distributed optimizer, we use DeepSpeed ZeRO Stage-3 optimizer (Rajbhandari et al., 2020). 
    DeepSpeed使った方がbatch size大きくできる？


なぜ遅いのか？
    DeepSpeed optimization accelerator



## B Technical Description of the Training Pipeline
[1]To achieve efficient pretraining, we use large batch training (Keskar et al., 2017) with 1.6M tokens per batch. 

[1]We set microbatch size to the maximal that fits to the GPU memory and we do gradient accumulation to get large enough batch size (to eventually have 1.6M tokens per batch). 
[1]Since training sequences have variable length (which comes from the fact that molecules have different sizes), only a part of tokens contribute to the loss so we make sure we have at least 1.6M such “enabled” tokens. 

[1] We use learning rate warmup of 2000 steps, followed by cosine annealing of the learning rate. 
[1] The maximal learning rate during pretraining is 10−3 regardless of the model size. 
[1] We use gradient clipping with the maximal grad norm of 1.0. 

[1] We use AdamW optimizer (Loshchilov & Hutter, 2019) with a weight decay factor of 10−2. 

[1] To use more performant tensor cores, we train with mixed precision where computation is done within the bfloat16 datatype. 

[] We train the model for 1 epoch only. 

([] The amount of tokens in the dataset is 42B for the version without explicit hydrogens and 90B tokens for the version with explicit hydrogens. )


The pretraining takes around 55k optimization steps over 36 hours on one compute node with 8 A6000 GPUs. 
The total size of the Uni-Mol pretraining dataset is around 150GB.