import numpy as np

class SliceManager():
    def __init__(self, slice_size=640, overlap_ratio=0.25, fill=65000):
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.fill = fill

    # input: array[h, w] int range[0, 65000]
    def padded_slice(self, img:np.ndarray):
        # padding -> orig_box, padded_img, padded_h, padded_w
        orig_h, orig_w = img.shape[-2:]
        orig_box =[0, 0, orig_h, orig_w]
        padding_h = 0 if orig_h > 640 else 640 - orig_h
        padding_w = 0 if orig_w > 640 else 640 - orig_w
        if padding_h > 0 or padding_w > 0:
            st_h = int(padding_h / 2)
            st_w = int(padding_w / 2)
            orig_box = [st_h, st_w, st_h+orig_h, st_w + orig_w]
            bg = np.ones((orig_h+padding_h, orig_w+padding_w)) * self.fill
            bg[orig_box[0]:orig_box[2], orig_box[1]:orig_box[3]] = img
            img = bg
        padded_h = orig_h + padding_h
        padded_w = orig_w + padding_w
        # slice -> n_slice_h, n_slice_w, slices
        slice_h_sts = list(range(0, padded_h-self.slice_size, int(self.slice_size * (1-self.overlap_ratio))))
        slice_h_sts.append(padded_h - self.slice_size)
        slice_w_sts = list(range(0, padded_w-self.slice_size, int(self.slice_size * (1-self.overlap_ratio))))
        slice_w_sts.append(padded_w - self.slice_size)
        n_slice_h = len(slice_h_sts)
        n_slice_w = len(slice_w_sts)
        slices = []
        for h_st in slice_h_sts:
            for w_st in slice_w_sts:
                slices.append(img[h_st:h_st+self.slice_size, w_st:w_st+self.slice_size])
        return slices, padded_h, padded_w, orig_box, n_slice_h, n_slice_w

    # input: pred_slices list[array[h, w]], float,range[0,1]
    def merged_crop(self, pred_slices, padded_h, padded_w, n_slice_h, n_slice_w, orig_box):
        bg = np.zeros((padded_h, padded_w))
        cnt_map = np.zeros_like(bg)
        # merge -> padded_pred
        assert len(pred_slices) == n_slice_h * n_slice_w
        h_st = w_st = ind = 0
        h_gap = int(self.slice_size * (1-self.overlap_ratio))
        w_gap = int(self.slice_size * (1-self.overlap_ratio))
        for i in range(n_slice_h):
            for j in range(n_slice_w):
                h_end = h_st + self.slice_size
                if h_end > padded_h:
                    h_st = padded_h - self.slice_size
                    h_end = padded_h
                w_end = w_st + self.slice_size
                if w_end > padded_w:
                    w_st = padded_w - self.slice_size
                    w_end = padded_w
                bg[h_st:h_end, w_st:w_end] = pred_slices[ind]
                cnt_map[h_st:h_end, w_st:w_end] += 1
                ind += 1
                w_st += w_gap
            h_st += h_gap
        bg = bg / cnt_map
        # crop -> orig_pred
        return bg[orig_box[0]:orig_box[2], orig_box[1]:orig_box[3]]
    


if __name__ == "__main__":
    slice_manager = SliceManager()
    test_img = np.ones((800, 400))
    slices, padded_h, padded_w, orig_box, n_slice_h, n_slice_w = slice_manager.padded_slice(test_img)
    slice_manager.merged_crop(slices, padded_h, padded_w, n_slice_h, n_slice_w, orig_box)