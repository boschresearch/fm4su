# Predicting the Road Ahead: A Knowledge Graph Based Foundation Model for Scene Understanding in Autonomous Driving

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication [Predicting the Road Ahead: A Knowledge Graph Based Foundation Model for Scene Understanding in Autonomous Driving](https://link.springer.com/chapter/10.1007/978-3-031-94575-5_7). It will neither be maintained nor monitored in any way.


## Requirements
1.	Clone the repository
```bash
git clone <repo-url>
cd <path-to-repo>/scene-understanding-foundation-model
```

2.	Create (if needed) and activate your Conda environment

```
conda create --name <env-name> python=3.8 -y

# Activate the environment
conda activate <env-name>
```

3.	Install the Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Extraction
```bash
cd data_extract
python KGextract.py configs/config_20x11.yaml credentials.yaml
```



## Experiments
```bash
cd T5-base
python T5_base.py configs/next_scene.yaml
```
Configuration: Modify ./configs/next_scene.yaml as needed.
Notes on T5 Configuration
- Task switch: Setting next_scene: True enables next-scene prediction; False enables multi-mask prediction. One setting takes precedence.
- Coverage matrix: Avoid overly large coverage ranges. The T5 script can predict up to the central 100 regions; adjust your matrix so that the ego vehicle remains centered and the width is an odd number.

## License
FM4SU is open-sourced under the AGPL-3.0 license. See the LICENSE file for details.