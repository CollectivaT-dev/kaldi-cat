#!/usr/bin/python3.8

import os
import re
from argparse import ArgumentParser, Namespace
import csv
import os
import re
import string
from unicodedata import normalize

import pandas as pd

from pythainlp.tokenize import newmm


def run_parser() -> Namespace:
    """Run argument parser"""
    parser = ArgumentParser()
    #parser.add_argument("--labels-path", type=str, required=True, help="Path to labels directory")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data root")
    parser.add_argument("--cv-path", type=str, required=True, help="Path to commonvoice corpus")
    parser.add_argument("--lexicon-path", type=str, required=True, help="Path to prepared lexicon")
    parser.add_argument("--phonemes-path", type=str, required=True, help="Path to prepared phoneme set")
    return parser.parse_args()


def format_df(df: pd.DataFrame, data_path: str, set_name: str, commonvoice_root: str, sr: int = 16000) -> None:
    """Format train/dev/test dataframe and stored in data root"""
    df = df[["path", "sentence"]]
    set_path = "{data_path}/{set_name}".format(data_path=data_path, set_name=set_name)
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    wav_scp = open("{set_path}/wav.scp".format(set_path=set_path), "w")
    utt2spk = open("{set_path}/utt2spk".format(set_path=set_path), "w")
    spk2utt = open("{set_path}/spk2utt".format(set_path=set_path), "w")
    text = open("{set_path}/text".format(set_path=set_path), "w")
    for i, (path, sent) in df.sort_values("path").iterrows():
        # # tokenize sentence with newmm
        # tokenized_sent = " ".join(newmm.segment(sent.replace(".", "")))
        # tokenized_sent = re.sub(r" +", " ", tokenized_sent)

        # clean sentence
        tokenized_sent = clean_line(sent)
        
        # write files to data/[train,dev,test]
        f_id = path.replace(".wav", "").replace(".mp3", "")
        wav_scp.write("{f_id} sox {commonvoice_root}/clips/{path} -t wav -r {sr} -c 1 -b 16 - |\n".format(f_id=f_id, commonvoice_root=commonvoice_root, path=path, sr=sr))
        utt2spk.write("{f_id} {f_id}\n".format(f_id=f_id))  # we wont specify spk id here
        spk2utt.write("{f_id} {f_id}\n".format(f_id=f_id))
        text.write("{f_id} {tokenized_sent}\n".format(f_id=f_id, tokenized_sent=tokenized_sent))
    wav_scp.close()
    utt2spk.close()
    spk2utt.close()
    text.close()


# normalize apostrophes, some we will keep
fix_apos = str.maketrans("`‘’", "'''")
# anything else we convert to space, and will squash multiples later
# this will catch things like hyphens where we don't want to concatenate words
all_but_apos = "".join(i for i in string.punctuation if i != "'")
all_but_apos += "–—“”"
clean_punc = str.maketrans(all_but_apos, (" " * len(all_but_apos)))
# keep only apostrophes between word chars => abbreviations
clean_apos = re.compile(r"(\W)'(\W)|'(\W)|(\W)'|^'|'$")
squash_space = re.compile(r"\s{2,}")
# chars not handled by unicodedata.normalize because not compositions
bad_chars = {'Æ': 'AE', 'Ð': 'D', 'Ø': 'O', 'Þ': 'TH', 'Œ': 'OE',
             'æ': 'ae', 'ð': 'd', 'ø': 'o', 'þ': 'th', 'œ': 'oe',
             'ß': 'ss', 'ƒ': 'f'}
clean_chars = str.maketrans(bad_chars)

def clean_line(textin):
    line = textin
    line = line.translate(fix_apos)
    line = line.translate(clean_punc)
    line = re.sub(clean_apos, r"\1 \2", line)
    line = re.sub(squash_space, r" ", line)
    line = line.strip(' ')
    # normalize unicode characters to remove accents etc.
    line = line.translate(clean_chars)
    line = normalize('NFD', line).encode('UTF-8', 'ignore')
    line = line.decode('UTF-8')
    line = line.lower()

    return line

def prepare_lexicon(data_path: str, source_lexicon_path: str, source_phones_path: str) -> None:
    """Prepare data/local/lang directory"""

    #TODO: Check if training data has words that are not in dictionary. 
    # with open("{data_path}/train/text".format(data_path=data_path), "r") as f:
    #     train_data = [" ".join(line.split(" ")[1:]).strip() for line in f.readlines()]
    # words = sorted(set([w for sent in train_data for w in sent.split(" ")]))
    
    lexicon = ["!SIL sil\n", "<UNK> spn\n"] + [line for line in open(source_lexicon_path, 'r').readlines()]
    nonsilence_phones = [g+"\n" for g in sorted(set([char[:-1] for char in open(source_phones_path, 'r').readlines()]))]
    optional_silence = ["sil\n"]
    silence_phones = ["sil\n", "spn\n"]
    
    if not os.path.exists("{data_path}/local/lang".format(data_path=data_path)):
        os.makedirs("{data_path}/local/lang".format(data_path=data_path))
    
    open("{data_path}/local/lang/lexicon.txt".format(data_path=data_path), "w").writelines(lexicon)
    open("{data_path}/local/lang/nonsilence_phones.txt".format(data_path=data_path), "w").writelines(nonsilence_phones)
    open("{data_path}/local/lang/optional_silence.txt".format(data_path=data_path), "w").writelines(optional_silence)
    open("{data_path}/local/lang/silence_phones.txt".format(data_path=data_path), "w").writelines(silence_phones)
    open("{data_path}/local/lang/extra_questions.txt".format(data_path=data_path), "w").writelines([])
    

def prepare_lexicon_naive(data_path: str) -> None:
    """Prepare data/local/lang directory"""
    with open("{data_path}/train/text".format(data_path=data_path), "r") as f:
        train_data = [" ".join(line.split(" ")[1:]).strip() for line in f.readlines()]
    words = sorted(set([w for sent in train_data for w in sent.split(" ")]))
    
    lexicon = ["!SIL sil\n", "<UNK> spn\n"] + [" ".join([word] + list(word))+"\n" for word in words]
    nonsilence_phones = [g+"\n" for g in sorted(set([char for word in words for char in word]))]
    optional_silence = ["sil\n"]
    silence_phones = ["sil\n", "spn\n"]
    
    if not os.path.exists("{data_path}/local/lang".format(data_path=data_path)):
        os.makedirs("{data_path}/local/lang".format(data_path=data_path))
    
    open("{data_path}/local/lang/lexicon.txt".format(data_path=data_path), "w").writelines(lexicon)
    open("{data_path}/local/lang/nonsilence_phones.txt".format(data_path=data_path), "w").writelines(nonsilence_phones)
    open("{data_path}/local/lang/optional_silence.txt".format(data_path=data_path), "w").writelines(optional_silence)
    open("{data_path}/local/lang/silence_phones.txt".format(data_path=data_path), "w").writelines(silence_phones)
    open("{data_path}/local/lang/extra_questions.txt".format(data_path=data_path), "w").writelines([])
    
    
def main(args: Namespace) -> None:
    train = pd.read_csv(args.cv_path+"/train.tsv", delimiter="\t")
    dev = pd.read_csv(args.cv_path+"/dev.tsv", delimiter="\t")
    test = pd.read_csv(args.cv_path+"/test.tsv", delimiter="\t")

    format_df(train, args.data_path, "train", args.cv_path)
    format_df(dev, args.data_path, "dev", args.cv_path)
    format_df(test, args.data_path, "test", args.cv_path)

    #prepare_lexicon_naive(args.data_path)
    prepare_lexicon(args.data_path, args.lexicon_path, args.phonemes_path)


if __name__ == "__main__":
    args = run_parser()
    main(args)

