# title

    

### 2. Training without Batch Normalization

* Ran for 19 epochs before early stopping
* Initial loss: 1.6406 → Final loss: 0.2938
* Best test loss: 0.6745 (Epoch 10)
* Signs of overfitting after e		poch 10 (train loss keeps decreasing while test loss increases)

### 3. Training with Batch Normalization

* Ran for 16 epochs
* Initial loss: 1.1711 → Final loss: 0.1311
* Best test loss: 0.5416 (Epoch 7)
* Faster convergence compared to non-batch normalized version

### Key Observations

1. **Convergence Speed** :

* Batch normalization converges faster (lower loss in early epochs)
* Initial loss with BN: 1.1711 vs without BN: 1.6406

1. **Overfitting** :

* Both models show signs of overfitting
* Without BN: starts around epoch 10
* With BN: starts around epoch 7

1. **Model Performance** :

* Batch normalization achieves better overall test loss
* Best test loss with BN: 0.5416
* Best test loss without BN: 0.6745
