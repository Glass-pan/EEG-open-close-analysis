import mne
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

subject = 1
runs = [1,2]

files = mne.datasets.eegbci.load_data(subject, runs)

raw_open = mne.io.read_raw_edf(files[0], preload=True)
raw_close = mne.io.read_raw_edf(files[1], preload=True)

standard_channels = [
"FC5","FC3","FC1","FCz","FC2","FC4","FC6",
"C5","C3","C1","Cz","C2","C4","C6",
"CP5","CP3","CP1","CPz","CP2","CP4","CP6",
"Fp1","Fpz","Fp2",
"AF7","AF3","AFz","AF4","AF8",
"F7","F5","F3","F1","Fz","F2","F4","F6","F8",
"FT7","FT8",
"T7","T8","T9","T10",
"TP7","TP8",
"P7","P5","P3","P1","Pz","P2","P4","P6","P8",
"PO7","PO3","POz","PO4","PO8",
"O1","Oz","O2","Iz"
]

raw_open.rename_channels(dict(zip(raw_open.ch_names, standard_channels)))
raw_close.rename_channels(dict(zip(raw_close.ch_names, standard_channels)))

raw_open.set_montage("standard_1005")
raw_close.set_montage("standard_1005")


def band_power(raw, fmin, fmax):
    psd = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=1024
    )
    return psd.get_data().mean(axis=1)


bands = {
"Theta":(4,7),
"Alpha":(8,12),
"Beta":(13,30)
}

fig, axes = plt.subplots(1,3,figsize=(15,5))

for i,(name,(fmin,fmax)) in enumerate(bands.items()):

    open_power = band_power(raw_open,fmin,fmax)
    close_power = band_power(raw_close,fmin,fmax)

    diff = np.log(open_power) - np.log(close_power)

    # ★ここが重要：バンドごとにスケール調整
    if name == "Alpha":
        vlim = (-5.0, 5.0)   # αは差が小さい → 強調
    elif name == "Theta":
        vlim = (-1, 1)
    else:
        vlim = (-1.0, 1.0)

    picks = mne.pick_types(raw_open.info,eeg=True)
    info = mne.pick_info(raw_open.info,picks)

    im, _ = mne.viz.plot_topomap(
        diff[picks],
        info,
        axes=axes[i],
        cmap="RdBu_r",
        vlim=vlim,
        contours=0,
        show=False
    )

    axes[i].set_title(name)

    plt.colorbar(im, ax=axes[i], fraction=0.046)

plt.suptitle("Open - Closed Brain Maps (θ / α / β)", fontsize=16)
plt.tight_layout()
plt.show()
