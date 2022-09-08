# . morph 

cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ../..

python3 exec.py demo google-demo \
    settings/google.json  \
    --paths paths/google-demo.json  \
    --hparams hparams/nn/multitask_vocab600.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_attn_cat_large_V.json   \
    --reset
cd -