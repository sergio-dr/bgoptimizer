import numpy as np
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

# TODO: only use statistics() with a "verbose" argument?

class ImageProcessor:
    
    config_defaults = {
        'downscaling_factor': 8, 
        'downscaling_func': 'median',
        'delinearization_quantile': 0.95,
    }    


    def __init__(self, config={}):
        self.config = {**self.config_defaults, **config}
        self.im_fr_lin_median, self.im_fr_nl_median = None, None
        self.pedestal, self.scale, self.bg_val = None, None, None


    def fit_transform(self, im_fr_lin): # delinearize+downscale
        # Preprocessing uses a copy
        im = im_fr_lin.copy()

        # Ignore zero values (real data has some pedestal) by converting to NaN
        self._num_to_nan(im)

        _, self.im_fr_lin_median, _ = self.statistics(im_fr_lin, "Original")

        # Delinearize to stretch the background
        im = self._delinearize(im)
        _, self.im_fr_nl_median, _ = self.statistics(im, "Delinearized")

        # Downscale
        im = self.downscale(im)
        _ = self.statistics(im, "Downscaled")

        # Replace NaNs
        np.nan_to_num(im, copy=False)

        return im


    def inverse_transform(self, bg_fr_nl): # linearize (input already at full res)
        bg_fr_lin = self._linearize(bg_fr_nl)
        _ = self.statistics(bg_fr_lin, "Bg (linear)")
        return bg_fr_lin


    def subtract_safe(self, im_fr_lin, bg_fr_lin):
        # subtract background model from the original image
        im_final = im_fr_lin - bg_fr_lin
        im_final_min, _, _ = self.statistics(im_final, "Subtracted")

        # Visualize out of range (negative, really) values
        #plt.figure(figsize=(16,10))
        #plt.imshow(-np.squeeze(im_final).clip(-1,0))
        #plt.title("Pixels with negative value after subtraction")

        # Apply pedestal so the final image has roughly the same median value
        # as the original.
        im_final -= im_final_min # Remove?
        im_final += self.im_fr_lin_median

        # Avoid clipping the highlights, it should not happen normally
        if im_final.max() > 1.0:
            im_final /= im_final.max()

        return im_final     


    def downscale(self, data):
        downscaling_func = {
            'median': np.nanmedian,
            'mean': np.nanmean
        }[self.config['downscaling_func']]

        block_sz = (self.config['downscaling_factor'], self.config['downscaling_factor'], 1)
        return block_reduce(data, block_size=block_sz, func=downscaling_func)


    # https://pixinsight.com/forum/index.php?threads/auto-histogram-settings-to-replicate-auto-stf.8205/#post-55143
    # donde (sea m=bg_val):
    #   mtf(0, m, r) = 0
    #   mtf(m, m, r) = bg_target_val
    #   mtf(1, m, r) = 1
    @staticmethod
    def _mtf(x, bg_val, bg_target_val=0.5):
        m, r = bg_val, 1/bg_target_val
        return ( (1-m)*x ) / ( (1-r*m)*x + (r-1)*m ) 


    @staticmethod
    def _imtf(y, bg_val, bg_target_val=0.5):
        m, r = bg_val, 1/bg_target_val
        return ImageProcessor._mtf(y, 1-m, (r-1)/r)


    # Mirroring np.nan_to_num
    @staticmethod
    def _num_to_nan(data, num=0.0):
        data[np.isclose(data, num)] = np.nan


    def _delinearize(self, data, bg_target_val=0.25):
        # Subtract pedestal
        self.pedestal = np.nanmin(data, axis=(0,1))
        data -= self.pedestal
        
        # Scale to [0,1] range, mapping the given quantile (instead of the max) to 1.0.
        # This blows out the highlights, but we are interested in the background!
        self.scale = np.nanquantile(data, q=self.config['delinearization_quantile'], axis=(0,1)) 
        data /= self.scale
        data = data.clip(0.0, 1.0)

        # Estimate background value
        self.bg_val = np.nanmedian(data, axis=(0, 1))

        return self._mtf(data, self.bg_val, bg_target_val)


    def _linearize(self, data, bg_target_val=0.25):
        data = self._imtf(data, self.bg_val, bg_target_val)
        data *= self.scale
        data += self.pedestal
        return data


    @staticmethod
    def statistics(im, title=""):
        im_min, im_med, im_max = np.nanmin(im, axis=(0,1)), np.nanmedian(im, axis=(0,1)), np.nanmax(im, axis=(0,1))
        print(f"[{title.ljust(12)}] Min / Median / Max = {float(im_min):.4f} / {float(im_med):.4f} / {float(im_max):.4f}", end='')
        print("  (!)" if (im_min < 0).any() or (im_max > 1).any() else "")
        return im_min, im_med, im_max


    @staticmethod
    def plot_image_hist(im, title=""):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 1]})

        _ = ax0.imshow(np.squeeze(im), vmin=0, vmax=1)

        color = 'rgb' if im.shape[-1] == 3 else ['black']
        for c in range(im.shape[-1]):
            _ = ax1.hist(im[..., c].ravel(), bins=100, alpha=0.25, color=color[c])

        fig.suptitle(title, fontsize=24)
        fig.tight_layout()
        fig.show()