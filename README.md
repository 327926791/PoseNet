pytorch implementation for PoseNet based on ICCV 2015 paper PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]
Download the Cambridge Landmarks King's College dataset from https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip

Download pretrained PoseNet weight: places-googlenet.pickle and put under pretrained_models

run: python train.py, available options: --epochs --learning_rate --batch_size --save_freq --data_dir

run: python test.py for evaluate, available options: --epochs --batch_size --batch_size
