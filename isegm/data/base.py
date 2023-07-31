import random
import pickle
import numpy as np
import cv2
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample


class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1,
                 copy_paste_prob=0.,
                 image_mix_prob=0.):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.copy_paste_prob = copy_paste_prob
        self.image_mix_prob = image_mix_prob
        self.to_tensor = transforms.ToTensor()

        self.dataset_samples = None

    def get_random_sample(self):
        # randomly select a target image
        target_index = random.randrange(0, len(self.dataset_samples))
        target_sample = self.get_sample(target_index)
        target_sample = self.augment_sample(target_sample)
        target_sample.remove_small_objects(self.min_object_area)
        return target_sample


    def __getitem__(self, index):
        if self.samples_precomputed_scores is not None:
            index = np.random.choice(self.samples_precomputed_scores['indices'],
                                     p=self.samples_precomputed_scores['probs'])
        else:
            if self.epoch_len > 0:
                index = random.randrange(0, len(self.dataset_samples))

        sample = self.get_sample(index)
        sample = self.augment_sample(sample)
        sample.remove_small_objects(self.min_object_area)

        if self.copy_paste_prob > 0 or self.image_mix_prob > 0:
            if self.copy_paste_prob > 0:
                # 1. C&P the sample object to another image
                if np.random.rand() < self.copy_paste_prob:
                    # choose an object from sample and put it on target_sample's image
                    target_sample = self.get_random_sample()
                    target_image = target_sample.image  # get image from target_sample

                    # select an object from sample
                    self.points_sampler.sample_object(sample)
                    obj_image = sample.image
                    obj_mask = self.points_sampler.selected_mask[0].astype(int) 
                    obj_indx = obj_mask > 0  # copy object from sample and paste into target_image

                    # apply alpha mixing, put obj on the target image
                    alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                    target_image[obj_indx] = cv2.addWeighted(
                        obj_image, alpha,
                        target_image, 1 - alpha,
                        0
                    )[obj_indx]

                    # construct a new sample
                    sample = DSample(target_image, obj_mask,
                                    objects_ids=[1], sample_id=index)

                # 2. C&P irrelevant object on sample image (do not fully cover the sample object)
                if np.random.rand() < self.copy_paste_prob:
                    # select an object from sample
                    self.points_sampler.sample_object(sample)
                    obj_image = sample.image
                    obj_mask = self.points_sampler.selected_mask[0].astype(int) 
                    obj_indx = obj_mask > 0  # copy object from sample and paste into target_image

                    for _ in range(5):  # at most try n times
                        target_sample = self.get_random_sample()
                        # choose an irrelevant object from the target_sample
                        self.points_sampler.sample_object(target_sample)
                        irr_obj_image = target_sample.image
                        irr_obj_mask = self.points_sampler.selected_mask[0].astype(int)
                        irr_obj_indx = irr_obj_mask > 0

                        if (obj_indx & ~irr_obj_indx).sum() <= 20:  # almost fully convered, retry
                            continue

                        # put mask on it by chance
                        choice = np.random.randint(0, 3)
                        if choice == 0:
                            # put irr_obj in obj image
                            obj_image[irr_obj_indx] = irr_obj_image[irr_obj_indx]
                            obj_mask[irr_obj_indx] = 0  # mask out the irr_obj
                        elif choice == 1:
                            # put irr_obj in obj image
                            obj_image[irr_obj_indx] = irr_obj_image[irr_obj_indx]
                            obj_mask = (obj_mask | irr_obj_mask).astype(int)
                        else:
                            # alpha mixing
                            alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                            obj_image[irr_obj_indx] = cv2.addWeighted(
                                obj_image, alpha,
                                irr_obj_image, 1 - alpha,
                                0
                            )[irr_obj_indx]

                        # construct a new sample
                        sample = DSample(obj_image, obj_mask,
                                        objects_ids=[1], sample_id=index)
                        break

            if np.random.rand() < self.image_mix_prob:
                # randomly select a target image
                target_sample = self.get_random_sample()
                # apply random image mixing augmentation
                # choose an random image and mix with the current target_sample
                alpha = np.random.rand() / 2 + 0.5  # transform alpha into [0.5, 1]
                sample.image = cv2.addWeighted(
                    sample.image, alpha,
                    target_sample.image, 1 - alpha,
                    0
                )

        self.points_sampler.sample_object(sample)
        points = np.array(self.points_sampler.sample_points())

        # give random order for random points
        # except the first click
        points_pos = points[:, 2] == 100
        indx = np.arange(points_pos.sum()) + 2
        np.random.shuffle(indx)
        points[points_pos, 2] = indx

        mask = self.points_sampler.selected_mask

        output = {
            'images': self.to_tensor(sample.image),
            'points': points.astype(np.float32),
            'instances': mask
        }

        if self.with_image_info:
            output['image_info'] = sample.sample_id

        return output

    def augment_sample(self, sample) -> DSample:
        if self.augmentator is None:
            return sample

        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores
