cd ./models_/pointmlp/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../pointops
python setup.py install
cd ../subsampling 
python setup.py install
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
cd ../../../../../SMPLer_X/main/transformer_utils/
pip install -v -e .
cd ../../../models_/pam/pam_mmcv
python setup.py install
cd ../pam_mmpose
python setup.py install
pip install numpy==1.23.0