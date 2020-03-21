#!/bin/bash

tb_root=$1
port=$2
tb_command="tensorboard --logdir ${tb_root} --port ${port}"

tmux new-session -d -s "tb_session"
tmux send -t tb_session "$tb_command" ENTER
