# medXGAN: Visual Explanations for Medical Classifiers through a Generative Latent Space
The brain tumor data can be downloaded from here: https://drive.google.com/file/d/1RK1whFxkEchuPdlw9MsURLSE-4L9deE2/view?usp=sharing. The COVID experiments rely on an in-house dataset, but the setup is similar.
### Training
Download the repo, and unzip the data into the code directory. To train medXGAN, run: 

```bash
python3 Main.py --root_dir {path-to-brain-data} --classifier_root_dir {path-to-classifier-weights}
```
### Visualizations
See the Jupyter Notebooks for examples on running visualizations: https://drive.google.com/drive/folders/1rEqJvJ2Qz9uKa8MN2lQDNuDG_eTjt3Tv?usp=sharing
