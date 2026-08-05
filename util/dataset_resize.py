from util.utils import *
from torchvision.transforms.functional import InterpolationMode
import os
from torchvision.transforms.functional import resize  # type: ignore
from torch.utils.data import Dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



class TrainSetLoader_Re_Pad(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None,
                 sample_list=None, augment=True):
        super(TrainSetLoader_Re_Pad).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        if sample_list is None:
            with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
                sample_list = f.read().splitlines()
        self.train_list = list(sample_list)
        self.augment = augment
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        self._cache = {}
        self._preload()

    def _preload(self):
        self._cache = {idx: self._load_item(idx) for idx in range(len(self.train_list))}

    def _load_item(self, idx):
        try:
            img = Image.open(
                (self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//', '/')).convert(
                'I')  # read image base on version ”I“
            # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + '.png').replace('//','/'))
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open(
                (self.dataset_dir + '/images/' + self.train_list[idx] + '.bmp').replace('//', '/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + '.bmp').replace('//', '/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)  # convert PIL to numpy  and  normalize
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        # print(self.train_list[idx])
        # img_patch, mask_patch = random_crop(img, mask, self.patch_size,
        #                                     pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256
        h,w = img.shape
        # resize
        target_size = PadImg_len(h, w, self.patch_size)
        # Resize tensors directly.  Converting normalized floats through PIL casts
        # them to uint8, which both destroys the normalization and makes Conv2d/
        # BCELoss fail at training time.
        input_image = resize(
            torch.from_numpy(img).unsqueeze(0), target_size,
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ).squeeze(0).numpy()
        input_mask = resize(
            torch.from_numpy(mask).unsqueeze(0), target_size,
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0).numpy()

        return input_image.astype(np.float32), input_mask.astype(np.float32)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._load_item(idx)
        image, mask = self._cache[idx]
        if self.augment:
            image, mask = self.tranform(image, mask)
        image = preprocess(self.patch_size, torch.from_numpy(np.ascontiguousarray(image[None])).float())
        mask = preprocess(self.patch_size, torch.from_numpy(np.ascontiguousarray(mask[None])).float())
        return image, mask

    def __len__(self):
        return len(self.train_list)




class TestSetLoader_Re_Pad(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size_eva,
                 img_norm_cfg=None, sample_list=None):
        super(TestSetLoader_Re_Pad).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        self.patch_size_eva = patch_size_eva
        if sample_list is None:
            with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
                sample_list = f.read().splitlines()
        self.test_list = list(sample_list)
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self._cache = {}
        self._preload()

    def _preload(self):
        self._cache = {idx: self._load_item(idx) for idx in range(len(self.test_list))}

    def _load_item(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//', '/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//', '/')).convert(
                'I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//', '/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if img.shape!=mask.shape:
            print(self.test_list[idx])

        h, w = img.shape
        # resize
        target_size = PadImg_len(h, w, self.patch_size_eva)
        input_image = resize(
            torch.from_numpy(img).unsqueeze(0), target_size,
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ).squeeze(0).numpy()


        img, mask = input_image[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()
        # pad
        img = preprocess(self.patch_size_eva, img)

        # input_mask = np.array(resize(to_pil_image(mask), target_size))
        # mask = preprocess(self.patch_size_eva, mask)

        return img, mask, target_size, [h, w], self.test_list[idx]

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._load_item(idx)
        return self._cache[idx]

    def __len__(self):
        return len(self.test_list)




class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
