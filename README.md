# Token_net
This is the code for our 2023 Research

## Requirements
Python == 3.8

Pytorch == 1.11

dgl == 0.9

CUDA == 10.2

graph-walker == 1.0.6

## Usage

You can run each command in "commands.txt".

You could change the hyper-parameters of NAGphormer if necessary.

Due to the space limitation, we only provide several small datasets in the "dataset" folder.

For small-scale datasets, you can download them from https://docs.dgl.ai/tutorials/blitz/index.html.

For large-scale datasets, you can download them from https://github.com/wzfhaha/GRAND-plus.

## large-scale datasets download
```
pip install gdown
gdown --id 1G9Wn1OaqMYpkNmbOESYUFrDgzo0Be0-L -O dataset/aminer.zip
gdown --id 1KauMd-AJXyD6KQQnf4vySjRZEOgWQYvx -O dataset/reddit.zip
gdown --id 1uItY1AGywFv4nSSFpqBaTEUoDn3w414B -O dataset/Amazon2M.zip
gdown --id 1VKHFQfRXkkVShE6d4hA9dImXZalz49qa -O dataset/mag_scholar_c.npz
```