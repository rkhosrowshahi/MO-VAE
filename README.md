# MO-VAE
![Alt sample generated](outputs/figures/samples_generated.png)\
A multi-objective representation learning in variational autoencoder to stabilize the gradients calculated by breaking down the evidence lower bound (ELBO) loss function into two objectives.
Our proposed method, named aligned multi-objective VAE (Aligned-MO-VAE), inspired by the multi-task learning, integrates two objectives known as reconstrion error and KL divergence simulatenously to optimize parameters with a gradient system that has zero conflict with others.
## Installation
* Close the repository
```
git clone https://github.com/rkhosroshahli/MO-VAE
cd MO-VAE
```
* Let's create virtual environment (venv) in the project and install necessary packages using ```pip```.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->
## Usage
* Train VAE on CIFAR10 with aligned-mtl
```
python main.py --dataset cifar10 --epochs 200 --agg aligned_mtl --optimizer adamw --lr 0.001 --save_freq 10 --latent_dim 128 --hidden_dims 32 64 128 --objs mse_sum kl_mean --output_activation tanh --normalize --use_wandb --wandb_name "128d mse_sum + kl_mean tanh amtl"
```
* Train VAE on CIFAR10 with UPGrad
```
python main.py --dataset cifar10 --epochs 200 --agg upgrad --optimizer adamw --lr 0.001 --save_freq 10 --latent_dim 128 --hidden_dims 32 64 128 --objs mse_sum kl_mean --output_activation tanh --normalize --use_wandb --wandb_name "128d mse_sum + kl_mean tanh upgrad"
```
* Train VQ-VAE on CIFAR10
```
python main.py --dataset cifar10  --arch vq_vae --epochs 200 --agg aligned_mtl --optimizer adamw --lr 0.001 --save_freq 10 --use_wandb --wandb_name "vq_vae emb_dim=64 num_emb=512 w/o activation" --embedding_dim 64 --num_embedding 512 --objs mse_sum --output_activation none --normalize
```
* Train VQ-VAE on CIFAR10 with tanh output activation and normalize dataset
```
python main.py --dataset cifar10  --arch vq_vae --epochs 200 --agg aligned_mtl --optimizer adamw --lr 0.001 --save_freq 10 --use_wandb --wandb_name "vq_vae emb_dim=64 num_emb=512 tanh" --embedding_dim 64 --num_embedding 512 --objs mse_mean --output_activation tanh --normalize --beta 0.25
```
<!-- CONTACT -->
## Contact
Rasa Khosrowshahli - rkhosrowshahli@brocku.ca