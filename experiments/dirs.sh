mkdir clean_tpugraphs_v2/npz

mkdir clean_tpugraphs_v2/npz/layout
mkdir clean_tpugraphs_v2/npz/layout/xla
mkdir clean_tpugraphs_v2/npz/layout/nlp
mkdir clean_tpugraphs_v2/npz/layout/nlp/random
mkdir clean_tpugraphs_v2/npz/layout/nlp/default

echo "MAKING NLP DIRS"
mkdir clean_tpugraphs_v2/npz/layout/nlp/random/train
mkdir clean_tpugraphs_v2/npz/layout/nlp/default/train

mkdir clean_tpugraphs_v2/npz/layout/nlp/random/test
mkdir clean_tpugraphs_v2/npz/layout/nlp/default/test

mkdir clean_tpugraphs_v2/npz/layout/nlp/default/valid
mkdir clean_tpugraphs_v2/npz/layout/nlp/random/valid

echo "MAKING XLA DIRS"
mkdir clean_tpugraphs_v2/npz/layout/xla/random/train
mkdir clean_tpugraphs_v2/npz/layout/xla/default/train

mkdir clean_tpugraphs_v2/npz/layout/xla/random/test
mkdir clean_tpugraphs_v2/npz/layout/xla/default/test

mkdir clean_tpugraphs_v2/npz/layout/xla/default/valid
mkdir clean_tpugraphs_v2/npz/layout/xla/random/valid


echo "MAKING TILE DIRS"
mkdir clean_tpugraphs_v2/npz/tile
mkdir clean_tpugraphs_v2/npz/tile/xla
mkdir clean_tpugraphs_v2/npz/tile/xla/train
mkdir clean_tpugraphs_v2/npz/tile/xla/test
mkdir clean_tpugraphs_v2/npz/tile/xla/valid
