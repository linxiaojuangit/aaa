import numpy as np
import matplotlib.pyplot as plt

loss = []
filename = './Record_RMSE_epoch.txt'
with open(filename, 'r') as fid:
    while True:
        aline = fid.readline()
        if not aline:
            break
        loss.append(float(aline.split(',')[4]))
loss = np.array(loss, dtype=np.float)
print('The minval of loss: ', np.min(loss))
print('The epoch of loss: ', np.argmin(loss))

plt.figure()
plt.plot(np.arange(len(loss)), loss)
plt.plot(np.argmin(loss), np.min(loss), 'r+')
plt.show()
