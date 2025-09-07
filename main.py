from src.data_utils import load_spotify_data
import src.model as md
import src.gardient as gd
import src.losses as loss
import src.optim as opt
import numpy as np
import matplotlib.pylab as plt

X_train, y_train, X_test, y_test = load_spotify_data("data/SpotifyFeatures.csv")

y = y_train
X = X_train

#hyperparameters
lr = 0.0001  # Reduced learning rate for better stability
beta = 0.9
lambda_ = 0.001
epochs = 200
momentum = True

#start vals
d = X_train.shape[1]
w, b = md.start_vals(d)  # weights and bias
vw = np.zeros_like(w)  # momentum term for weights
vb = 0.0 

#model

losses = []
old_loss = 0.0  
for e in range(epochs): 
    indices = np.random.permutation(len(X_train)) #shuffle data
    loss_value = 0.0

    for i in indices: #stochastic gradient descent

        x_i = X_train[i:i+1] # keep 2D shape
        y_i = y[i:i+1] # keep 2D shape
        
        z = md.z_calc(x_i, w, b) # linear combination
        y_hat = md.sigmoid(z)               # predicted probabilities

        loss_value += loss.cross_entropy(y_i, y_hat, eps=1e-12) # accumulate loss

        dw, db = gd.gradient(y_i, y_hat, x_i, lambda_, w) # gradients

        w, b, vw, vb = opt.gradient_step(w, b, dw, db, lr, beta, vw, vb, momentum) # update weights
    loss_mean = loss_value/len(X_train) # average loss over epoch

    loss_test = (e+1, loss_mean) # store epoch and loss
    losses.append(loss_test) 
    

    if abs(old_loss - loss_value) < 1e-6:  # Check for convergence based on small change in loss
        print(f"Converged at epoch {e}")
        break
    old_loss = loss_value  # Update old_loss after the check

print(f"loss = {loss_mean}")
print(f"w = {w[0]}x1 + {w[1]}x2 + {b}") 
epochs, loss_values = zip(*losses)

w_final = w
b_final = b

# TRAIN
z_tr   = md.z_calc(X_train, w_final, b_final) 
p_tr   = md.sigmoid(z_tr)                 # predicted probabilities
yhat_tr = (p_tr >= 0.5).astype(int)       # predicted classes

# TEST
z_te   = md.z_calc(X_test, w_final, b_final)  
p_te   = md.sigmoid(z_te)                
yhat_te = (p_te >= 0.5).astype(int)

acc_tr = (yhat_tr == y_train).mean()
acc_te = (yhat_te == y_test).mean()
print(f"train acc = {acc_tr:.3f} | test acc = {acc_te:.3f}") 

TP = int(((yhat_te==1)&(y_test==1)).sum()) 
TN = int(((yhat_te==0)&(y_test==0)).sum()) 
FP = int(((yhat_te==1)&(y_test==0)).sum())
FN = int(((yhat_te==0)&(y_test==1)).sum()) 
print(f"Confusion matrix (test):\nTP={TP}  FP={FP}\nFN={FN}  TN={TN}") 


plt.plot(epochs, loss_values, label="Train loss") 
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.legend()
params_text = (
    f"Learning rate: {lr}\n"
    f"Momentum: {beta}\n"
    f"Epochs: {len(epochs)}"
)

plt.gca().text( 
    0.95, 0.95, params_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.show()

