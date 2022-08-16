cd ../..
python3.6 exec.py official reddit_slim_train_batch  \
    settings/reddit-slim-batch.json  \
    --paths paths/reddit-slim-batch.json  \
    --hparams hparams/nn/multitask_vocab600.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_attn_cat_large_V.json   \
    --reset
cd -