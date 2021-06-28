echo "Downloading raw trajectory data (71MB) and our preprocessed data (2.6GB)"
cd data
gdown https://drive.google.com/uc?id=1i9BxkrktNqt4TWkzwF1S7qf_ASgG894b -O full_2.1.0_pp.tar.gz
tar -zxf full_2.1.0_pp.tar.gz
gdown https://drive.google.com/uc?id=1D4VMc30BSo2fSmoVFOowB8Px1NBVAZpT -O full_2.1.0.zip
unzip -q full_2.1.0.zip
cd ..

echo "Downloading pretrained Mask-RCNN models"
gdown https://drive.google.com/uc?id=11p9x-TXSaLkdvWQGg6vKlLXlAo08vW4a -O models/detector/mrcnn_object.pth
gdown https://drive.google.com/uc?id=1CiIGJxhH6z9Up5Yqqj-XLdSqdD-uAiD7 -O models/detector/mrcnn_receptacle.pth

echo "Downloading the trained HiTUT model"
gdown https://drive.google.com/uc?id=1ykVUiXOrFTqHIdaOsyYEKeqZQzMngPR0 -O exp/Jan27-roberta-mix.zip
unzip exp/Jan27-roberta-mix.zip -d exp