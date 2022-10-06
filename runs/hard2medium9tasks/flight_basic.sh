cd ../..
python3.6 exec.py official flight_basic  \
    settings/flight_basic.json  \
    --paths paths/flight_basic.json  \
    --hparams hparams/nn/multitask_vocab600.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/flight_basic.json   \
    --reset
cd -


