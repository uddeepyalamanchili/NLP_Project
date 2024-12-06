
set -e
set -x
COMMIT=388c520be21752cacb9fe3b1712038f32e0e9a5f
echo "Downloading Fairseq from https://github.com/pytorch/fairseq (rev:$COMMIT)"
wget https://github.com/shamilcm/fairseq-py/archive/$COMMIT.zip
unzip $COMMIT.zip
rm $COMMIT.zip
mv fairseq-py-$COMMIT fairseq-py

COMMIT=b9453d5a211fc8f90fb25a584b39d4784f8de716
echo "Downloading n-best reranker from https://github.com/nusnlp/nbest-reranker (rev: $COMMIT)"
wget https://github.com/nusnlp/nbest-reranker/archive/$COMMIT.zip
unzip $COMMIT.zip
rm $COMMIT.zip
mv nbest-reranker-$COMMIT nbest-reranker
#git clone https://github.com/nusnlp/nbest-reranker/

COMMIT=ec5c7b009c409e72b5ef65a77c1a846546f14847
echo "Downloading Subword NMT from https://github.com/rsennrich/subword-nmt (rev: $COMMIT)"
wget https://github.com/rsennrich/subword-nmt/archive/$COMMIT.zip
unzip $COMMIT.zip
rm $COMMIT.zip
mv subword-nmt-$COMMIT subword-nmt

patch -p0 -i changes.patch