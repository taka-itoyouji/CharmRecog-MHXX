#!/bin/bash

dataset=20220430

# pngdir=~/database/monsterhunter/XX/goseki_png/20220311/test
pngdir=~/database/monsterhunter/XX/goseki_png/20220311
pngdir=~/database/monsterhunter/XX/goseki_png/test

outdir=../out/$dataset
tooldir=../tool

# スキルリストからスキルポイントを取得
# cat ~/database/monsterhunter/XX/skill_list.csv | grep ,+ | cut -d ',' -f 1 > ../out/skill_list.txt

#main
python $tooldir/main.py --outdir $outdir --pngdir $pngdir

# dataset=test

# outdir=../out/$data

# # test
# python $tooldir/main.py --outdir $outdir