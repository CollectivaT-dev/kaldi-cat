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

phonemes="../dict/ca/phonemes.txt"
lexicon="../dict/ca/lexicon.txt"
textcorpus="../corpus/CA_OpenSubtitles_clean.txt" #This will be added on top of train/dev text to build LM text corpus

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
  echo ">> 0: download and untar"

  mkdir -p $cv_base_path
  local/download_and_untar_CV.sh $cv_base_path $lang

  mkdir -p $pp_base_path
  local/download_and_untar_PP.sh --remove-archive $pp_base_path
fi

if [ $stage -le 1 ]; then
  echo ">> 1: prepare datasets"
  echo "python local/prepare_data.py --data-path $data_path --cv-path $cv_path --pp-path $pp_base_path --phonemes-path $phonemes --lexicon-path $lexicon --subset $subset"
  python local/prepare_data.py --data-path $data_path --cv-path $cv_path --pp-path $pp_base_path --phonemes-path $phonemes --lexicon-path $lexicon  --subset $subset || { echo "Fail running local/prepare_cv.py"; exit 1; }
fi

if [ $stage -le 2 ]; then
  echo ">> 2a: validate prepared data"
  for part in train dev cv_test pp_test; do
    utils/validate_data_dir.sh --no-feats data/$part || { echo "Fail validating $part"; exit 1; }
  done

echo ">> 2b: prepare LM and format to G.fst"
  utils/prepare_lang.sh data/local/lang '<UNK>' data/local data/lang

  local/prepare_lm.sh --order $lm_order --textcorpus $textcorpus || { echo "Fail preparing LM"; exit 1; }
  local/format_data.sh || { echo "Fail creating G.fst"; exit 1; }
fi

if [ $stage -le 3 ]; then
  echo ">> 3: create MFCC feats (make_mfcc_pitch)"
  for part in train dev pp_test cv_test; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $njobs data/$part exp/make_mfcc/$part $mfccdir || { echo "Error make MFCC features"; exit 1; }
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir || { echo "Error computing CMVN"; exit 1; }
  done
fi

# train monophone
if [ $stage -le 4 ]; then
  echo ">> 4: train monophone"
  steps/train_mono.sh --boost-silence 1.25 --nj $njobs --cmd "$train_cmd" \
    data/train data/lang exp/mono || { echo "Error training mono"; exit 1; };
  (
    utils/mkgraph.sh data/lang exp/mono exp/mono/graph || { echo "Error making graph for mono"; exit 1; }
    for testset in dev; do
      steps/decode.sh --nj $njobs --cmd "$decode_cmd" exp/mono/graph \
        data/$testset exp/mono/decode_$testset || { echo "Error decoding mono"; exit 1; }
    done
  )&
  steps/align_si.sh --boost-silence 1.25 --nj $njobs --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali_train || { echo "Error aligning mono"; exit 1; }
fi

# train delta + delta-delta triphone
if [ $stage -le 5 ]; then
  echo ">>5: train delta + delta-delta triphone"
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train data/lang exp/mono_ali_train exp/tri1 || { echo "Error training delta tri1"; exit 1; }

  # decode tri1
  (
    utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || { echo "Error making graph for tri1"; exit 1; }
    for testset in dev; do
      steps/decode.sh --nj $njobs --cmd "$decode_cmd" exp/tri1/graph \
        data/$testset exp/tri1/decode_$testset || { echo "Error decoding tri1"; exit 1; }
    done
  )&

  steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali_train || { echo "Error aligning tri1"; exit 1; }
fi

# LDA+MLLT
if [ $stage -le 6 ]; then
  echo ">>6a: train LDA+MLLT"
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
      data/train data/lang exp/tri1_ali_train exp/tri2b || { echo "Error training tri2b (LDA+MLLT)"; exit 1; }

  echo ">>6b: decode LDA+MLTT"
  utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph || { echo "Error making graph for tri2b"; exit 1; }
  (
    for testset in dev; do
    steps/decode.sh --nj $njobs --cmd "$decode_cmd" exp/tri2b/graph \
      data/$testset exp/tri2b/decode_$testset || { echo "Error decoding tri2b"; exit 1; }
    done
  )&

  echo ">>6c: align using tri2b"
  steps/align_si.sh --nj $njobs --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/tri2b exp/tri2b_ali_train || { echo "Error aligning tri2b"; exit 1; }
fi

# tri3b, LDA+MLLT+SAT
if [ $stage -le 7 ]; then
  echo ">>7a: train LDA+MLLT"
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang exp/tri2b_ali_train exp/tri3b || { echo "Error training tri3b (LDA+MLLT+SAT)"; exit 1; }

  echo ">>7b: decode using the tri3b model"
  (
    utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph || { echo "Error making graph for tri3b"; exit 1; }
    for testset in dev; do
      steps/decode_fmllr.sh --nj $njobs --cmd "$decode_cmd" \
        exp/tri3b/graph data/$testset exp/tri3b/decode_$testset || { echo "Error decoding tri3b"; exit 1; }
    done
  )&
fi

if [ $stage -le 8 ]; then
  echo ">>8a: align utts in the full training set using the tri3b model"
  steps/align_fmllr.sh --nj $njobs --cmd "$train_cmd" \
    data/train data/lang \
    exp/tri3b exp/tri3b_ali_train || { echo "Error aligning FMLLR for tri4b"; exit 1; }

  echo ">>8b: train another LDA+MLLT+SAT system on the entire training set"
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train data/lang \
    exp/tri3b_ali_train exp/tri4b || { echo "Error training tri4b"; exit 1; }

  echo ">>8c: decode using the tri4b model"
  (
    utils/mkgraph.sh data/lang exp/tri4b exp/tri4b/graph || { echo "Error making graph for tri4b"; exit 1; }
    for testset in dev; do
      steps/decode_fmllr.sh --nj $njobs --cmd "$decode_cmd" \
        exp/tri4b/graph data/$testset \
        exp/tri4b/decode_$testset || { echo "Error decoding tri4b"; exit 1; }
    done
  )&
fi

# train a chain model
if [ $stage -le 9 ]; then
  echo ">>9: train a chain model"
  local/chain/run_tdnn.sh --stage 0
fi

# wait for jobs to finish
wait

# print best WERs
bash RESULT.sh
