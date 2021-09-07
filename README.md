# Gender Classification
This project is to classify the Gender from a 12-Lead ECG Signal.
## Author

[Felix Tempel](mailto:felixtempel95@hotmail.de?subject=[GitLab]%20Gender%20Classification)
	
## Technologies
Project is created with the Pytorch framework using a 34-Layer ResNet Model.
For Optimizing the SGD-Optimizer is used.

	
## Setup
To run this project, clone the repository and install the requirements:

```
$ git clone 
$ pip install -r requirements.txt
```

### Data
The data for training the model can be found under the following link: 
https://physionetchallenges.github.io/2020/

For the classification **three** data sets have been used which are:

**CPSC2018 training set, 6.877 recordings**

**PTB-XL electrocardiography Database, 21.837 recordings**

**Georgia 12-Lead ECG Challenge Database, 10.344 recordings**

The data is sampled at 500Hz and consists of a .hea and .mat file.

The model was trained on the healthy ECG Files.

The data files have to be downloaded into a folder of your desire - 
the path has to be given as an argument when starting the project

If you have Google Colab account you can run the following command:

```
./get_data.sh
```


The folder structure for the data should be the following:

```
datafolder/
 |
 +-- chin_database
 |  |
 |  +-- files
 |   
 +-- ptb_databse
 |  |
 |  +-- files
 |    
 +-- georgia_database
 |  |  
 |  +-- files
 |    
 +-- mit_database
 |  |  
 |  +-- files
...
```

### Docker

To run this project with Docker build the container in the following way:

```
git clone
cd proj
docker build -t docker_gender_classification .
docker run --mount type=bind, source="data_root", target="data_root" --gpus all --shm-size=8g -it docker_gender_classification
```

### Terminal

To run the project from the Terminal:
```
cd gender_classification
python3 main.py -h

positional arguments:
  <command>    train, hypertrain or test
  <database>   Dataframe file location on your system
  <files>      File location on your system

optional arguments:
  -h, --help   show this help message and exit
  --ckpt CKPT  Path to model weights
  --ex EX      Additional save name
```
Example:
```
python3 main.py test df_mix_op.csv ./src/datafolder
```

### Visualization

To visualize the model a tensorboard logger is implemented - to view the parameters:

```
tensorboard --logdir=path/to/tensorboard/file
```




