import imgaug.augmenters as iaa
import torch
import numpy as np

from contextlib import contextmanager

# heavily copy from https://github.com/shrubb/latent-pose-reenactment
class ParametricAugmenter:
    def is_empty(self):
        return not self.seq and not self.shift_seq

    def __init__(self, use_pixelwise_augs,use_affine_scale,use_affine_shift):
        

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        total_augs = []

        if use_pixelwise_augs:
            pixelwise_augs = [
                iaa.SomeOf((0, 5),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 0.25), n_segments=(150, 200))),
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(1, 3)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(1, 3)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.BlendAlphaSimplexNoise(
                                   iaa.EdgeDetect(alpha=(0.0, 0.15)),
                               ),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=False),
                               # add gaussian noise to images
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToSaturation((-20, 20)),  # change hue and saturation
                               iaa.JpegCompression((70, 99)),

                               iaa.Multiply((0.5, 1.5), per_channel=False),

                               iaa.OneOf([
                                   iaa.LinearContrast((0.75, 1.25), per_channel=False),
                                   iaa.SigmoidContrast(cutoff=0.5, gain=(3.0, 11.0))
                               ]),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.15)),
                               # move pixels locally around (with random strengths)
                           ],
                           random_order=True
                           )
            ]
            total_augs.extend(pixelwise_augs)
        affine_augs_scale = []
        if use_affine_scale:
            affine_augs_scale = [sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                order=[1],  # use  bilinear interpolation (fast)
                mode=["reflect"]
            ))]
            # total_augs.extend(affine_augs_scale)

        if use_affine_shift:
            affine_augs_shift = [sometimes(iaa.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                order=[1],  # use bilinear interpolation (fast)
                mode=["reflect"]
            ))]
        else:
            affine_augs_shift = []

        self.shift_seq = iaa.Sequential(affine_augs_shift)
        self.seq = iaa.Sequential(total_augs, random_order=True)
        self.scale_seq = iaa.Sequential(affine_augs_scale)

    def tensor2image(self, image, norm = 255.0):
        return (np.expand_dims(image.permute(1, 2, 0).numpy(), 0) * norm)

    def image2tensor(self, image, norm = 255.0):
        image = image.astype(np.float32) / norm
        image = torch.tensor(np.squeeze(image,0)).permute(2, 0, 1)
        return image


    def augment_tensor(self, image):
        if self.seq or self.shift_seq:
            image = self.tensor2image(image).astype(np.uint8)
            image = self.seq(images=image)
            image = self.shift_seq(images=image,)
            image = self.image2tensor(image)

        return image

    def augment_quadra(self, image1, image2,mask1,mask2):
        if self.seq or self.shift_seq:
            image1 = self.tensor2image(image1).astype(np.uint8)
            mask2 = self.tensor2image(mask2).astype(np.uint8)
            image1 = self.seq(images=image1,)
            if self.scale_seq:
                
                scale_seq_deterministic = self.scale_seq.to_deterministic()
                image1 = scale_seq_deterministic(images=image1)
                mask1 = scale_seq_deterministic(images=mask1)
            if self.shift_seq:
                image2 = self.tensor2image(image2).astype(np.uint8)
                shift_seq_deterministic = self.shift_seq.to_deterministic()
                image1 = shift_seq_deterministic(images=image1,)
                image2 = shift_seq_deterministic(images=image2)
                mask1 = shift_seq_deterministic(images=mask1)
                mask2 = shift_seq_deterministic(images=mask2)
                image2 = self.image2tensor(image2)

            image1 = self.image2tensor(image1)
            mask2 = self.image2tensor(mask2)
        mask1 = self.image2tensor(mask1)
        return image1, image2,mask1,mask2

    def augment_double(self, image,mask):
        if self.seq or self.shift_seq:
            image = self.tensor2image(image).astype(np.uint8)
            mask = self.tensor2image(mask).astype(np.uint8)
            image = self.seq(images=image,)
            if self.scale_seq:
                
                scale_seq_deterministic = self.scale_seq.to_deterministic()
                image = scale_seq_deterministic(images=image)
                mask = scale_seq_deterministic(images=mask)
            if self.shift_seq:
                shift_seq_deterministic = self.shift_seq.to_deterministic()
                image = shift_seq_deterministic(images=image,)
                mask = shift_seq_deterministic(images=mask)
              

            image = self.image2tensor(image)
            mask = self.image2tensor(mask)
      
        return image,mask

    @contextmanager
    def deterministic_(self, seed):
        """
        A context manager to pre-define the random state of all augmentations.

        seed:
            `int`
        """
        # Backup the random states
        old_seq = self.seq.deepcopy()
        old_shift_seq = self.shift_seq.deepcopy()
        self.seq.seed_(seed)
        self.shift_seq.seed_(seed)
        yield
        # Restore the backed up random states
        self.seq = old_seq
        self.shift_seq = old_shift_seq
