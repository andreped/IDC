import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
import h5py
from sklearn.metrics import classification_report
from tqdm import tqdm

# name for model
name = "model_IDC_classification_17_07"

# paths
data_path = "C:/Users/andrep/Documents/Projects/DP/IDC/data/"
save_model_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/saved_models/"
history_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/history/"
datasets_path = "C:/Users/andrep/Documents/Projects/DP/IDC/output/datasets/"

# get test set
f = h5py.File(datasets_path + "dataset_" + name + '.h5', 'r')
X = np.array(f["X_test"])
Y = np.array(f["Y_test"])
f.close()

# load model
model = load_model(save_model_path + name + ".h5", compile=False)

gts = []
preds = []

for i in tqdm(range(X.shape[0])):
	X_curr = X[i]
	Y_curr = Y[i]

	pred = model.predict(np.expand_dims(X_curr, axis=0))

	gt = np.argmax(Y_curr)
	pred = np.argmax(pred)

	gts.append(gt)
	preds.append(pred)


report = classification_report(gts, preds, labels=[0, 1], digits=4)
print(report)

