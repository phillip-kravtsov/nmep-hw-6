mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
tar -xvf resnet_v2_50_2017_04_14.tar.gz
mv resnet_v2_50.ckpt ${CHECKPOINT_DIR}
rm resnet_v2_50_2017_04_14.tar.gz
