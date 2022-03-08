#!/bin/bash

# Commonvoice-th kaldi's recipe
# Modify from kaldi's commonvoice recipe

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# default paths
#docker (needs edit)
#cv_path="/mnt/cv-corpus-7.0-2021-07-21"
#labels_path="/mnt/labels"

#local
corpora_base_path="/home/$USER/LargeDrive/corpora"
cv_base_path="$corpora_base_path/commonvoice"
pp_base_path="$corpora_base_path/PP"

data_path="data"
mfccdir=mfcc

njobs=$(nproc)  # num jobs, default as num CPU
lm_order=3  # lm order

stage=0
lang=ca
subset=0

. ./utils/parse_options.sh || exit 1;

cv_path=$cv_base_path/$lang

if [ $stage -le 0 ]; then
  mkdir -p $cv_base_path
  local/download_and_untar.sh $cv_base_path $lang
  mkdir -p $pp_base_path
  local/download_and_untar_PP.sh --remove-archive $pp_base_path
fi

if [ $stage -le 1 ]; then
  # prepare dataset
  echo "python local/prepare_cv.py --data-path $data_path --cv-path $cv_path --phonemes-path ../dict/ca/phonemes.txt --lexicon-path ../dict/ca/lexicon.txt --subset $subset"
  python local/prepare_cv.py --data-path $data_path --cv-path $cv_path --phonemes-path ../dict/ca/phonemes.txt --lexicon-path ../dict/ca/lexicon.txt  --subset $subset || { echo "Fail running local/prepare_cv.py"; exit 1; }
fi

if [ $stage -le 2 ]; then
  # validate prepared data
  #for part in train dev dev_unique test test_unique; do
  for part in train dev test; do
    utils/validate_data_dir.sh --no-feats data/$part || { echo "Fail validating $part"; exit 1; }
  done

  utils/prepare_lang.sh data/local/lang '<UNK>' data/local data/lang

  # prepare LM and format to G.fst
  local/prepare_lm.sh --order $lm_order || { echo "Fail preparing LM"; exit 1; }
  local/format_data.sh || { echo "Fail creating G.fst"; exit 1; }
fi

# Extract MFCC features
if [ $stage -le 3 ]; then
  for task in train; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$task exp/make_mfcc/$task $mfcc
    steps/compute_cmvn_stats.sh data/$task exp/make_mfcc/$task $mfcc
  done
fi

# Train GMM models
if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/mono

  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali

  steps/train_lda_mllt.sh  --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono_ali exp/tri1

  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    2500 15000 data/train data/lang exp/tri1_ali exp/tri2

  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    2500 20000 data/train data/lang exp/tri2_ali exp/tri3

  steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali
fi

# Train TDNN model
if [ $stage -le 5 ]; then
  local/chain/run_tdnn.sh
fi

# Decode
if [ $stage -le 6 ]; then

  utils/format_lm.sh data/lang data/local/lm/lm_tgsmall.arpa.gz data/local/lang/lexicon.txt data/lang_test
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test exp/chain/tdnn exp/chain/tdnn/graph
  utils/build_const_arpa_lm.sh data/local/lm/lm_tgmed.arpa.gz \
    data/lang data/lang_test_rescore

  for task in test; do

    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$task exp/make_mfcc/$task $mfcc
    steps/compute_cmvn_stats.sh data/$task exp/make_mfcc/$task $mfcc

    steps/online/nnet2/extract_ivectors_online.sh --nj 10 \
        data/${task} exp/chain/extractor \
        exp/chain/ivectors_${task}

    steps/nnet3/decode.sh --cmd $decode_cmd --num-threads 10 --nj 1 \
         --beam 13.0 --max-active 7000 --lattice-beam 4.0 \
         --online-ivector-dir exp/chain/ivectors_${task} \
         --acwt 1.0 --post-decode-acwt 10.0 \
         exp/chain/tdnn/graph data/${task} exp/chain/tdnn/decode_${task}

    steps/lmrescore_const_arpa.sh data/lang_test data/lang_test_rescore \
        data/${task} exp/chain/tdnn/decode_${task} exp/chain/tdnn/decode_${task}_rescore
  done

  bash RESULTS.sh
fi
