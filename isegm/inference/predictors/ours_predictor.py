import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide, ResizeTrans
from isegm.utils.crop_local import map_point_in_bbox, get_focus_cropv1, get_focus_cropv2, get_object_crop, \
    get_click_crop

class NoRefinePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size=256,
                 focus_crop_r=1.4,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.crop_l = infer_size


        self.transforms = []
        self.transforms.append(ResizeTrans(self.crop_l))
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None
        self.global_roi = None

        if self.with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    # def get_prediction(self, clicker, prev_mask=None, **kwargs):
    #     clicks_list = clicker.get_clicks()
    #     click = clicks_list[-1]
    #     last_y, last_x = click.coords[0], click.coords[1]
    #     self.last_y = last_y
    #     self.last_x = last_x
    #
    #     if self.click_models is not None:
    #         model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
    #         if model_indx != self.model_indx:
    #             self.model_indx = model_indx
    #             self.net = self.click_models[model_indx]
    #
    #     input_image = self.original_image
    #     if prev_mask is None:
    #         prev_mask = self.prev_prediction
    #     if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
    #         input_image = torch.cat((input_image, prev_mask), dim=1)
    #
    #     image_nd, clicks_lists, is_image_changed = self.apply_transforms(
    #         input_image, [clicks_list]
    #     )
    #
    #     pred_logits, feature = self._get_prediction(image_nd, clicks_lists, is_image_changed)
    #     prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
    #                                size=image_nd.size()[2:])
    #
    #     for t in reversed(self.transforms):
    #         prediction = t.inv_transform(prediction)
    #
    #     self.prev_prediction = prediction
    #     return prediction.cpu().numpy()[0, 0]
    def get_prediction(self, clicker, prev_mask=None, on_cascade=False, **kwargs):
        clicks_list = clicker.get_clicks()
        self.cascade_step = 2
        self.cascade_clicks = 1
        self.cascade_adaptive = True

        if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
            for i in range(self.cascade_step):
                prediction = self.get_prediction(clicker, None, True)
                if self.cascade_adaptive and prev_mask is not None:
                    diff_num = (
                            (prediction > 0.49) != (prev_mask > 0.49)
                    ).sum()
                    if diff_num <= 20:
                        return prediction
                prev_mask = prediction
            return prediction

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]


    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output = self.net(image_nd, points_nd)
        return output['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1, y2, x1, x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg = len(clicks_list) - num_pos
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [], []
        for click in clicks_list:
            flag, y, x, index = click.is_positive, click.coords[0], click.coords[1], 0
            y, x = map_point_in_bbox(y, x, y1, y2, x1, x2, self.crop_l)
            if flag:
                pos_clicks.append((y, x, index))
            else:
                neg_clicks.append((y, x, index))

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')

