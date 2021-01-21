# Cross-Thought
Cross-Thought for Sentence Encoder Pre-training (https://arxiv.org/abs/2010.03652)
## Install
```
git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/fairseq.git
git clone https://github.com/ngoyal2707/Megatron-LM.git fairseq/fairseq/model_parallel/megatron
cp src/transformer_sentence_encoder.py fairseq/fairseq/modules/transformer_sentence_encoder.py
pip install --editable ./fairseq
```
Overall, we only change one file 'transformer_sentence_encoder.py' from RoBERTa for Cross-Thought. All the data processing follows RoBERTa setting.

## Finetune
Please follow RoBERTa finetune (https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md) for data process

Cross-Thought checkpoint (https://drive.google.com/file/d/11lnZijWuRcPT07xiEO5NMhkXMfKopIb9/view?usp=sharing)
```
export CTPRETRAIN=False
TOTAL_NUM_UPDATES=113272  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=28318      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=32        # Batch size.
ROBERTA_PATH=/path/to/cross-thought-checkpoint.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train data/QQP-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 0 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0 --attention-dropout 0 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
```

## Pre-train
Please follow RoBERTa pre-train (https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md) for data process
```
export CTPRETRAIN=True
fairseq-train --fp16 $DATA_DIR --task masked_lm --criterion masked_lm     \
--arch roberta_base --sample-break-mode none --tokens-per-sample 32000 \
--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0    \
--lr-scheduler polynomial_decay --lr 0.0005 --warmup-updates 10000 \
--total-num-update 125000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01    \
--batch-size 1 --update-freq 16 --max-update 500000 --log-format simple --log-interval 1 \
--save-dir outputs/crossthought --save-interval-updates 5000
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

MIT
