from toolkit.utils import load_dataset
import pywt
import numpy as np
import matplotlib.pyplot as plt
_,ucr_dataset,_,_ = load_dataset(data_name="UCR", group_name="1", task_name="single")
data = ucr_dataset[0][:20000, 0]

# Wavelet transform
wavelet = "cmor1.5-1.0"
scales = np.arange(10, 200, 1)

coefs, freqs = pywt.cwt(data, scales, wavelet, 1)

plt.figure(figsize=(120, 10))
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(np.arange(len(data)), data)
axes[0].set_title("Original signal")
axes[1].contourf(np.arange(len(data)), freqs, np.abs(coefs))
axes[1].set_title("Wavelet coefficients")
plt.savefig("wavelet_analysis.png", dpi=300)
# plt.show()

