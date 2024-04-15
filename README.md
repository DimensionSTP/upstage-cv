# Basic DL pipeline for CV competition

## For Upstage CV competition

### Dataset
Upstage CV competition dataset(document image classification)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/upstage-cv.git
cd upstage-cv

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Model Hyper-Parameters Tuning

* end-to-end
```shell
python main.py mode=tune is_tuned=untuned num_trials={num_trials}
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={tuned or untuned} num_trials={num_trials}
```

### Prediction

* end-to-end
```shell
python main.py mode=predict is_tuned={tuned or untuned} epoch={ckpt epoch} submission_name={submission_name}
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__