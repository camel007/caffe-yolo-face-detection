#!/usr/bin/env sh
LOG=log/train-`data +%Y-%m-%d-%H-%M-%S`.log

./build/tools/caffe train --solver examples/yolo/yolo-face.solver --weights examples/yolo/yolo-face.caffemodel -gpu 0 2>&1 | tee $LOG

