cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ../..

python3.6 exec.py detection_demo detection \
    settings/detection.json  \
    --paths paths/detection.json  \
    --hparams hparams/nn/multitask_vocab600.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_attn_cat_large_V.json   \
    --reset
cd -