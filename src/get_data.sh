mkdir data
wget -O PhysioNetChallenge2020_Training_CPSC.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_CPSC.tar.gz/
wget -O PhysioNetChallenge2020_Training_PTB-XL.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_PTB-XL.tar.gz/
wget -O PhysioNetChallenge2020_Training_E.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_E.tar.gz/
tar xf PhysioNetChallenge2020_Training_CPSC.tar.gz -C ./data/chin_database
rm PhysioNetChallenge2020_Training_CPSC.tar.gz
tar xf PhysioNetChallenge2020_Training_PTB-XL.tar.gz -C ./data/ptb_database
rm PhysioNetChallenge2020_Training_PTB-XL.tar.gz
tar xf PhysioNetChallenge2020_Training_E.tar.gz -C ./data/georgia_database
rm PhysioNetChallenge2020_Training_E.tar.gz








