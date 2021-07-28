#Python read hdf5 boxes
#Author: Willem Elbers
#Date: 9 April 2020

import numpy as np;
import h5py;
from matplotlib import pyplot as plt;
import os;
import sys;

fname = sys.argv[1];
if (len(sys.argv) > 2):
	dsetname = sys.argv[2];
else:
	dsetname = "Field/Field";

f = h5py.File(fname, "r");
d = f[dsetname];

print("Box width:", d.shape);
print("Reading the box from disk.");

arr = np.array(d);

#Get rid of padding
#arr = arr[:,:,:-2];

#Plot a slice of the array
plt.imshow(arr[16]);plt.colorbar();plt.xlabel("z");plt.ylabel("y");plt.show();

print("");
print("Sum:\t", arr.sum());
print("Sigma:\t", np.sqrt(arr.var()));
