# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ModuleNotFoundError:
    pyspng = None

from torchvision import transforms
import torch.nn.functional as F

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        **ignore                # I am ignoring "resolution" key in training_loop, seems like a bug 
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):

        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        try:
            assert self.image_shape[1] == self.image_shape[2], "Niente"
        except:
            print(f'Irregular image of shape {self.image_shape}, {self.image_shape[1]} defines resolution')
            
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#Datasetobj that loads facies and Ip distributions
class FaciesSet(Dataset):

    def __init__(self,
                 path,
                 image_size,
                 image_depth,
                 **super_kwargs 
                 ):
        
        t = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomInvert(1),
                transforms.Normalize(0.5,0.5),
                transforms.RandomVerticalFlip(1),
                ])
        t2 = transforms.Compose([])
        self.fullsize = np.array([80,100])
        self.size = np.array(image_size)
        self.path = path
        try: 
            self.crop = self.size < self.fullsize #False if image_size[0] in [80,100,128] else True
        except:
            self.crop = None
        
        self.pad = (8 - (self.size % 8)) % 8 #8 is due to the network used by Karras et al., 2022
        
        if (self.pad>0).any(): 
            self.pad_tuple = (0, self.pad[-1], self.pad[-2], 0)
            t2.transforms.append(lambda x: F.pad(x, self.pad_tuple))
        
        self.len_data = len(os.listdir(self.path+'/Facies/'))
        self.transforms = [t,t2]
        
        self.image_depth = image_depth
        
        if self.image_depth>1:
            self.max_ip_reals=0 ; i=0; t = True
            while t==True:
                t = os.path.isfile(self.path+f'/Ip/0_{i}.pt')
                if t==True: self.max_ip_reals +=1
                i+=1
            ip1 = torch.load(self.path+f'/Ip/{np.random.randint(0,self.len_data)}_0.pt', weights_only=True)
            ip2 = torch.load(self.path+f'/Ip/{np.random.randint(0,self.len_data)}_0.pt', weights_only=True)

            self.ipmin = min(ip1.min(),ip2.min())
            self.ipmax = max(ip1.max(),ip2.max())
            
            ip1= ip1[:,:self.size[0],:self.size[1]]
            
        name = os.path.splitext(os.path.basename(self.path))[0]
        raw_shape = [self.len_data] + [self.image_depth] + list(self.size)
        
        #if image_size is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #    raise IOError('Image files do not match the specified resolution')
        
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    def __len__(self):
        return self.len_data
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        out_dict = torch.zeros(0) #this is just to integrate possible conditions in a second time
        out = self.transforms[0](PIL.Image.open(self.path+f'/Facies/{idx}.png'))[0,None]
        
        if self.image_depth > 1 :
            Ip = self.path+f'/Ip/{idx}_{np.random.randint(0,self.max_ip_reals)}.pt'
            Ip = torch.load(Ip, weights_only=True)
            Ip = 2*((Ip - self.ipmin) / (self.ipmax - self.ipmin))-1
            out = torch.cat((out, Ip), dim=0)
       
        if self.crop[0]: 
            idx = np.random.randint(0,80-self.size[0])
            out = out[:,idx:idx+self.size[0]]
        if self.crop[1]: 
            idx = np.random.randint(0,100-self.size[1])
            out = out[:,:,idx:idx+self.size[1]]
        
        out = self.transforms[1](out) #pad if necessary
            
        return out, out_dict

            
class FaciesSet_parse3D(Dataset):
    """
    Loads random slices from randomly selected TI volumes.

    Expected files:
        facies_TI0.npy ... facies_TI9.npy
        poro_TI0.npy   ... poro_TI9.npy

    Important:
    - Uses memory mapping (mmap_mode='r') to avoid loading full volumes in RAM.
    - Only the selected slice/window is read from disk.
    - Facies and poro always use the same TI index.
    """

    def __init__(self,
                 path,
                 image_size,
                 image_depth,
                 n_ti=20,
                 n_val=5,
                 transname = 'standard05',
                 **super_kwargs):

        self.path = path
        self.return_weights = False
        self.size = image_size
        self.image_depth = image_depth
        self.epsilon = 0.001
        self.n_ti = n_ti
        self.n_val = n_val
        self.facies_volumes = []
        self.poro_volumes = []

        self.facies_validation = []
        self.poro_validation = []
        
        assert image_depth == 2 #not handling 1 property now

        for i in range(n_ti):
            self.facies_volumes.append(
                np.load(f"{self.path}/facies_TI{i}.npy", mmap_mode='r'))

            self.poro_volumes.append(
                np.load(f"{self.path}/poro_TI{i}.npy", mmap_mode='r'))
            print(f'Loading real {i}')
        
        if n_val is not None:
            for i in range(n_val):
                self.facies_validation.append(
                    np.load(f"{self.path}/facies_val{i}.npy", mmap_mode='r'))

                self.poro_validation.append(
                    np.load(f"{self.path}/poro_val{i}.npy", mmap_mode='r'))
                print(f'Loading real {i}')
        
        else: 
            self.facies_validation = self.facies_volumes
            self.poro_validation = self.poro_volumes
            self.n_val = self.n_ti
            print('!! No validation data is being used ')
        
        vol_shape = self.facies_volumes[0].shape
        self.nz, self.ny, self.nx = vol_shape

        # ------------------------------------------------------------
        # Normalization stats
        # [channel, (min/max or mean/std), 1,1]
        # channel 0 = facies
        # channel 1 = poro
        # ------------------------------------------------------------
        self.min_max = np.zeros((image_depth, 2, 1, 1), dtype=np.float32)
        self.mean_std = np.zeros((image_depth, 2, 1, 1), dtype=np.float32)

        # ---------- FACIES ----------
        # keep your original fixed normalization
        self.min_max[0, 0] = min(v.min() for v in self.facies_volumes)
        self.min_max[0, 1] = max(v.max() for v in self.facies_volumes)
        
        
        # mean/std computed incrementally
        fac_means = [v.mean() for v in self.facies_volumes]
        fac_stds = [v.std() for v in self.facies_volumes]
        self.mean_std[0, 0] = np.mean(fac_means)
        self.mean_std[0, 1] = np.mean(fac_stds)

        # class weights from first TI only
        # f0 = self.facies_volumes[0]
        # self.f_classes, n = np.unique(f0, return_counts=True)
        # self.w = 1 / (n / f0.size)
        # self.w = self.w / self.w.sum()

        # ---------- PORO ----------
        self.min_max[1, 0] = min(v.min() for v in self.poro_volumes)
        self.min_max[1, 1] = max(v.max() for v in self.poro_volumes)

        # mean/std computed incrementally
        poro_means = [v.mean() for v in self.poro_volumes]
        poro_stds = [v.std() for v in self.poro_volumes]

        self.mean_std[1, 0] = np.mean(poro_means)
        self.mean_std[1, 1] = np.mean(poro_stds)

        # ------------------------------------------------------------
        # Transforms
        # ------------------------------------------------------------
        if transname== 'standard': transflist = [self.standard]
        elif transname== 'standard05': transflist = [self.standard05]
        elif transname=='minmax': transflist = [self.minmax]
        else: 
            assert transname==None ('Not a valid transformation')
            transflist = []
            
        self.pad = (8 - (np.array(self.size) % 8)) % 8

        if (self.pad > 0).any():
            self.pad_tuple = (0, self.pad[-1], self.pad[-2], 0)
            transflist.append(self.pad_zeros)

        self.transforms = transforms.Compose(transflist)

        # ------------------------------------------------------------
        # Dataset length estimate
        # ------------------------------------------------------------
        l1 = (self.nz - self.size[0]) + 1 #/ (self.size[0] / 2) + 1
        l2 = (self.nx - self.size[1]) + 1 #/ (self.size[1] / 2) + 1
        sideone = int(l1 * l2 ) # * self.ny)

        l1 = (self.ny - self.size[0]) + 1 #/ (self.size[0] / 2) + 1
        l2 = (self.nx - self.size[1]) + 1 #/ (self.size[1] / 2) + 1
        sidetwo = int(l1 * l2 ) #* self.nx)

        self.len_data = (sideone + sidetwo) * n_ti

        name = os.path.splitext(os.path.basename(self.path[:-1]))[0]

        raw_shape = [self.len_data*4] + [self.image_depth] + list(self.size)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __len__(self):
        return self.len_data

    def standard(self, x):
        return (x - self.mean_std[:, 0]) / self.mean_std[:, 1]

    def standard05(self, x):
        return (x - self.mean_std[:, 0]) / (self.mean_std[:, 1] * 2)

    def minmax(self, x):
        return 2 * (
            (x - self.min_max[:, 0]) /
            (self.min_max[:, 1] - self.min_max[:, 0])
        ) - 1

    def addepsilon(self, x):
        return x + np.random.normal(
            loc=0,
            scale=self.epsilon,
            size=x.shape
        )

    def pad_zeros(self, x):
        return F.pad(x, self.pad_tuple)

    def values_to_idx(self, x):
        return ((x + 1.5) / 1.0)

    def get_weight_map(self, x):
        idx = self.values_to_idx(x).astype(int)
        return self.w[idx].astype(np.float16)

    def random_selection(self, facies, poro):
        # Random crop coordinates
        z = np.random.randint(0, self.nz - self.size[0] + 1)
        x = np.random.randint(0, self.nx - self.size[1] + 1)
        y = np.random.randint(0, self.ny - self.size[1] + 1)

        nslice = np.random.randint(0, self.ny)

        # Random orientation
        if np.random.choice([0, 1]):
            facies_slice = facies[
                z:z + self.size[0],
                nslice,
                x:x + self.size[1]
            ]

            poro_slice = poro[
                z:z + self.size[0],
                nslice,
                x:x + self.size[1]
            ]

        else:
            facies_slice = facies[z:z + self.size[0],y:y + self.size[1],nslice]
            poro_slice = poro[z:z + self.size[0], y:y + self.size[1],nslice]

        # Stack channels
        # shape => [2,H,W]
        data = np.stack(
            [facies_slice, poro_slice],
            axis=0
        ).astype(np.float32)

        # normalize
        return data

    def getvalidation(self):
        idx = np.random.randint(0, self.n_val)
        facies = self.facies_validation[idx]
        poro = self.poro_validation[idx]
        out_dict = torch.zeros(0)
        # facies += np.random.normal(loc=0,scale=self.epsilon,size=facies.shape)

        return self.transforms(self.random_selection(facies, poro)), out_dict


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # RANDOMLY SELECT ONE TI
        ti = np.random.randint(0, self.n_ti)
        facies = self.facies_volumes[ti]
        poro = self.poro_volumes[ti]

        data = self.transforms(self.random_selection(facies, poro))

        # optional noise
        data[0] += np.random.normal(
            loc=0,
            scale=self.epsilon,
            size=data[0].shape
        )

        if self.return_weights:
            weight = self.get_weight_map(data[0])
            data = np.concatenate((data, weight[None, :]))
        
        out_dict = torch.zeros(0)
        return data, out_dict
#----------------------------------------------------------------------------    
    
class Geost_dataset(Dataset):
    
    def __init__(self,
                 path,
                 image_size,
                 image_depth,
                 **super_kwargs 
                 ):
        self.path = path
        self.size = image_size
        
        self.len_data = int(np.memmap(self.path+'/images.npy', mode='r', dtype='float32').shape[0]//np.prod(self.size))
        self.imgs = np.memmap(self.path+'/images.npy', mode='r', dtype='float32', shape=(self.len_data, image_size[0],image_size[1]))
        
        self.n_labels = np.memmap(self.path+'/labels.npy', mode='r', dtype='float32').shape[0]//self.len_data
        self.labels = np.memmap(self.path+'/labels.npy', mode='r', dtype='float32', shape=(self.len_data, self.n_labels))
        
        #let's perform normalization for each label
        self.minlbl = self.labels.min(0)
        self.maxlbl = self.labels.max(0)
        self.meanlbl = self.labels.mean(0)
        self.stdlbl = self.labels.std(0)
        
        assert not (np.isnan(self.labels).any() or np.isnan(self.imgs).any()), 'Invalid dataset: contains error'
        self.min = self.imgs.min(); self.max = self.imgs.max()
        self.meanimage = self.imgs.mean(); self.stdimage=self.imgs.std()
        self.size = image_size
        
        raw_shape = [self.len_data] + [1] + list(self.size)
        name = os.path.splitext(os.path.basename(self.path[:-1]))[0]

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        label = self.labels[idx]
        
        # label = (label - self.meanlbl[None,:]) / self.stdlbl[None,:]
        label = 2 * ((label - self.minlbl) / (self.maxlbl - self.minlbl)) - 1
        # image = (image - self.meanimage) / self.stdimage
        
        image = 2 * ((image - self.min) / (self.max - self.min)) - 1
        return image[None,:], label
    
    def _load_raw_image(self, raw_idx):
        return self.imgs[raw_idx]

    def _load_raw_labels(self):
        return self.labels
