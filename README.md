# DACG
This is the implementation of our paper entitled: "Dynamic attention-based CVAE-GAN for pedestrian trajectory prediction" <br>
![image text](https://github.com/zhourongleiden/DACG/blob/main/Framework.png)  <br>
FIgure: Overview of the proposed DACG model. Herein, the whole model consists of a generator and a discriminator. The generator is made up of five modules,
namely Feature Encoder, Social Aggregator, Mode Estimator, Goal Estimator, and Trajectory Decoder. Red lines indicate the processes that appear in the training
phase only.
## Contents
### folder
* results: storing the stepwise model training results <br>
* sgan: containing the DACG model <br>
* Trajectron: containing the dataset and dataloader <br>
### File
* main.py: the script for training <br>
* argument_parser_neighbor.py: the hyperparameters applied in the proposed DACG model <br>
## Model training
Please run `python main.py --dataset_name name` to train a DACG model from scratch <br>
(Please choose the "name" from {eth,hotel,univ,zara1,zara2})
