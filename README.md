# Bark Beetle Detection CNN
Run with default options using `python -m train_classifiers` in this directory.
Call with `-h` to list advanced options.
This script expects there to exist a directory `./dataset` with subdirectories `1` and `0`, respectively containing positive (bark beetle or beetle damage present) and negative (no beetle or beetle damage) examples.
A script for reproducing the reference dataset can be found at [this repository](https://github.com/sob505/bark-beetle-detection).
If saving training logs is desirable, consider doing something like `python -m train_classifiers \[options\] | tee ./output/log.txt`.
