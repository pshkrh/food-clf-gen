# Food Image Classification and Generation

## Food Image Classification

### Installation

The following packages are required:

- PyTorch
- Pandas
- scikit-learn
- matplotlib
- tqdm
- numpy
- kaggle

For PyTorch, please follow the instructions here (for pip or conda): https://pytorch.org/get-started/locally/

```shell
pip install numpy pandas matplotlib scikit-learn tqdm kaggle torch torchvision torchaudio
```

### Downloading the dataset

There are a few ways to obtain the Food-101 dataset. One is from the author's website [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
Another option is to download it from Kaggle [here](https://www.kaggle.com/datasets/dansbecker/food-101).

Alternatively, if you decide to run the Notebook, it already contains code to download the zip from either the dataset website or from kaggle. For the kaggle option via the Notebook, you will need a kaggle.json file in the same directory as the notebook. More info [here](https://www.kaggle.com/discussions/general/156610).

### Usage

The repository contains both Jupyter notebooks and Python scripts. The Jupyter notebooks have commands to execute any additional libraries that may be required for installation.
Included is also the experiment code that was run for training the Food-11 dataset as well in ```food11-training.ipynb```. All the notebooks can also be run in Google Colab as-is.


The python scripts have the same code as the notebooks, just more cleaned up and better organized. You will get the same results regardless of which option you use.

#### Training and Evaluation

This script will execute the entire training pipeline for all 5 models, both pretrained and non-pretrained. Two CSV files and pickle files will be saved at the end with the results on the test set and model metrics respectively. All the trained models will be stored in the ```Trained_Models``` folder in the same directory as the script.

```shell
python train_classifier.py
```

#### Inference

You will need to set the input image path in the ```main()``` function. The script will print out the predicted class.

```shell
python predict_image.py
```

#### Plotting

The jupyter notebook ```plots.ipynb``` contains the plotting code to generate plots for training and testing loss and accuracy for both pretrained and non-pretrained models.
You will need to specify the path to the pickle files for the pretrained and non-pretrained model metrics in order to generate the figures.

You can download the pickle files obtained from the training experiment here: https://drive.google.com/file/d/1cp-0ytNgz7hh4NbRV1jCWYsJ46QIf9YE/view?usp=sharing

## Pizza Image Generation

### Installation

The following packages are required:

- PyTorch
- Pillow
- matplotlib
- tqdm
- datasets
- diffusers
- accelerate

For up-to-date PyTorch instructions, please see here: https://pytorch.org/get-started/locally/

```shell
pip install torch torchvision torchaudio Pillow matplotlib tqdm datasets diffusers accelerate
```

### Usage

Similar to classification, both Jupyter notebooks ```train_diffusion.ipynb```, ```pizza_gen.ipynb``` and a script ```train_diffusion.py``` are provided.

### Training the Diffusion Model

This script will execute the training pipeline for the diffusion model for 100 epochs. The trained model will be saved as ```pizza-diffusion.pt``` in the same directory as the script or notebook.
The model directory containing the model config, weights, and logs will be created based on the configuration specified in ```TrainingConfig```.

Output images will be saved in the configured output directory at the path: ```output_dir/samples/```. Model configuration can be changed in the ```TrainingConfig``` class.

```shell
python train_diffusion.py
```

### Generating a Pizza Image

This script will generate a Pizza image. You will need to specify the model directory path in the code.

```shell
python generate_pizza_image.py
```

### Trained Model Download

The trained model is available at this link: https://drive.google.com/file/d/1FHntB14ZtXYLPNTH2J693Z0NEzxheiqe/view?usp=sharing