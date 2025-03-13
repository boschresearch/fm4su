# Towards a Foundation Model for Scene Understanding in Autonomous Driving

## Getting Started

Clone this repo:

Install the required packages by running:

```
module load conda

conda create --name <your env> # skip this line if environment already created

conda activate <your env>

cd <path to root file of scene-understanding-foundation-model>

module load proxy4server-access/2.0 # proxy access to install packages from internet

proxy_on

pip install -r requirements.txt
```

## Data extraction:

```
cd data_extract
```

The last line of KGextract.bsub indicates the program to run and the config paths, for example: 

```
python KGextract.py configs/config_20x11.yaml credentials.yaml
```

To change the configuration, please modify the file in ./configs directory accordingly

The credentials.yaml stores the login information for Stardog database, any change to it is set to be ignored by git, so safe to push.

After adjusting the config file and the bsub file, submit the job via:

```
bsub < KGextract.bsub
```

## T5 Model experimenting:

```
cd T5-base
```

Same logic with T5_base.bsub, no need of login credentials this time:
```
python T5_base.py configs/next_scene.yaml
```

To change the configuration, please modify the file in ./configs directory accordingly

After adjusting the config file and the bsub file, submit the job via:

```
bsub < T5_base.bsub
```

## T5 Configuration:

Note that by setting next_scene to True/False, it will switch the experiment to complete next scene prediction or multi-mask prediction. One will override the other.

Also, please make sure that the coverange range of the matrix won't be extremly large, since it would require significantly more compute. T5 can at most predict for the central 100 areas of an entry point, so better adjust the outer areas to be excluded in case of scene prediction. It is better to set the width of the matrix to an odd number since the ego vehicle is placed in the middle.
