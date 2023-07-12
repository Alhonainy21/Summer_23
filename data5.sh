#!/bin/bash
cd /
curl -L -o ddd https://www.dropbox.com/s/g430bmmuuc4nqrc/lung_generated.tar.gz?dl=0
tar -zxvf ddd
mv lung_generated lungg
cd lungg
find -name '._*' -delete
cd test
cd acatest
cd lung_aca5
mv *png /lungg/test/acatest
cd ..
rm -r lung*
cd ..
cd ntest
cd lung_n5
mv *png /lungg/test/ntest
cd ..
rm -r lung*
cd ..
cd stest
cd lung_scc5
mv *png /lungg/test/stest
cd ..
rm -r lung*

cd ../..
cd train
cd acatrain
cd lung_aca5
mv *png /lungg/train/acatrain
cd ..
rm -r lung*
cd ..
cd ntrain
cd lung_n5
mv *png /lungg/train/ntrain
cd ..
rm -r lung*
cd ..
cd strain
cd lung_scc5
mv *png /lungg/train/strain
cd ..
rm -r lung*
cd ../../..
