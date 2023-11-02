import numpy as np

def channeled_norm(image, mode='single'):

    # image [h, w], 0-65000
    assert mode in ['single', 'ms', 'sw']
    if mode == 'single':
        image = image / 65535
        image = image.astype(np.float32)
    elif mode == 'ms':
        ths = [511, 1023, 2047, 4095, 8191, 16383, 32767, 65535]
        channels = [(np.where(image >= th, th, 0) + (image < th) * image) / th for th in ths]
        image = np.concatenate(channels, axis=0).astype(np.float32)
    else:
        lows = [0, 256, 512, 1024, 2048, 4096, 8192, 16384]
        highs = [511, 1023, 2047, 4095, 8191, 16383, 32767, 65535]
        channels = [(np.where(image >= thr, thr-thl, 0) + (image < thr) * (image > thl) * (image - thl)) / (thr - thl)
                    for (thl, thr) in zip(lows, highs)]
        image = np.concatenate(channels, axis=0).astype(np.float32)
    return image