import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as T

import torchvision.models as torch_models
import numpy as np

import clip
from PIL import Image

class AppearanceLoss(torch.nn.Module):
    def __init__(self, args):
        super(AppearanceLoss, self).__init__()
        self.args = args

        args.texture_slw_weight = 0.0
        args.texture_ot_weight = 0.0
        args.texture_gram_weight = 0.0
        #Added loss types for CLIP
        args.texture_clipimg_weight = 0.0
        args.texture_cliptxt_weight = 0.0

        if args.appearance_loss_type == 'OT':
            args.texture_ot_weight = 1.0
        elif args.appearance_loss_type == 'SlW':
            args.texture_slw_weight = 1.0
        elif args.appearance_loss_type == 'Gram':
            args.texture_gram_weight = 1.0
        #Added loss types for CLIP
        elif args.appearance_loss_type == 'CLIPImg':
            args.texture_clipimg_weight = 1.0
        elif args.appearance_loss_type == 'CLIPTxt':
            args.texture_cliptxt_weight = 1.0

        self.slw_weight = args.texture_slw_weight
        self.ot_weight = args.texture_ot_weight
        self.gram_weight = args.texture_gram_weight
        #Added loss types for CLIP
        self.clipimg_weight = args.texture_clipimg_weight
        self.cliptxt_weight = args.texture_cliptxt_weight

        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}
        if self.slw_weight != 0:
            self.loss_mapper["SlW"] = SlicedWassersteinLoss(self.args)
            self.loss_weights["SlW"] = self.slw_weight

        if self.ot_weight != 0:
            self.loss_mapper["OT"] = OptimalTransportLoss(self.args)
            self.loss_weights["OT"] = self.ot_weight

        if self.gram_weight != 0:
            self.loss_mapper["Gram"] = GramLoss(self.args)
            self.loss_weights["Gram"] = self.gram_weight

        #Added loss types for CLIP
        if self.clipimg_weight != 0:
            self.loss_mapper["CLIPImg"] = ClipLossImgToImg(self.args)
            self.loss_weights["CLIPImg"] = self.clipimg_weight

        if self.cliptxt_weight != 0:
            self.loss_mapper["CLIPTxt"] = ClipLossTxtToImg(self.args)
            self.loss_weights["CLIPTxt"] = self.cliptxt_weight

    def update_losses_to_apply(self, epoch):
        pass

    def forward(self, input_dict, return_summary=True):
        loss = 0.0

        update_mask = input_dict['update_mask']
        target_image_list = input_dict['target_image_list']
        generated_image_list = input_dict['generated_image_list']
        for target_images, generated_images in zip(target_image_list, generated_image_list):
            b, c, h, w = generated_images.shape
            _, _, ht, wt = target_images.shape

            # Scale the images before feeding to VGG
            generated_images = (generated_images + 1.0) / 2.0 * update_mask[:, :3, :, :].to(self.args.DEVICE)
            target_images = (target_images + 1.0) / 2.0 * update_mask[:, :3, :, :].to(self.args.DEVICE)

            if h != ht or w != wt:
                target_images = TF.resize(target_images, size=(h, w))
            for loss_name in self.loss_mapper:
                loss_weight = self.loss_weights[loss_name]
                loss_func = self.loss_mapper[loss_name]
                if self.cliptxt_weight != 0:
                    target_text = input_dict['target_text']
                    loss += loss_weight * loss_func(target_text, generated_images)
                else:
                    loss += loss_weight * loss_func(target_images, generated_images)
        loss /= len(generated_image_list)
        return loss, None, None

#CLIP loss to compare 2 images. This was used to ensure that our CLIP Loss made sense by applying it to the same image used with regular appearance loss
class ClipLossImgToImg(torch.nn.Module):
    def __init__(self, args):
        super(ClipLossImgToImg, self).__init__()
        self.args = args
        self.model, self.preprocess = clip.load("ViT-B/32", device=args.DEVICE)
        

    def forward(self, target_images, generated_images):
        #Transform the image in the same way as described in CLIP's official github (https://github.com/openai/CLIP/blob/main/clip/clip.py)
        #This ensures the image can be properly understood by CLIP
        transforms = torch.nn.Sequential(
            T.Resize(224),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        )
        
        #pre-process the images so that CLIP can understand them
        with torch.no_grad():
            target_images_processed = transforms(target_images)
        generated_images_processed = transforms(generated_images)

        #Get the target and generated images' representation in CLIP's latent space, aka their CLIP features
        with torch.no_grad():
            target_features = self.model.encode_image(target_images_processed.to(self.args.DEVICE))
        generated_features = self.model.encode_image(generated_images_processed.to(self.args.DEVICE))

        # normalized features
        target_features = target_features / target_features.norm(dim=1, keepdim=True)
        generated_features = generated_features / generated_features.norm(dim=1, keepdim=True)

        # cosine similarity
        cos_per_generated = generated_features @ target_features.t()

        # shape = [global_batch_size, global_batch_size]
        return (- cos_per_generated.mean(axis = 1) + 1).sum()

#CLIP loss to compare an image with the target text.
class ClipLossTxtToImg(torch.nn.Module):
    def __init__(self, args):
        super(ClipLossTxtToImg, self).__init__()
        self.args = args
        self.model, self.preprocess = clip.load("ViT-B/32", device=args.DEVICE)
        

    def forward(self, target_text, generated_images):
        #Transform the image in the same way as described in CLIP's official github (https://github.com/openai/CLIP/blob/main/clip/clip.py)
        #This ensures the image can be properly understood by CLIP
        transforms = torch.nn.Sequential(
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        )
        
        #pre-process the images so that CLIP can understand them
        generated_images_processed = transforms(generated_images)

        #Get the target text and the generated images' representation in CLIP's latent space, aka their CLIP features
        with torch.no_grad():
            target_features = self.model.encode_text(clip.tokenize(target_text).to(self.args.DEVICE))
        generated_features = self.model.encode_image(generated_images_processed.to(self.args.DEVICE))

        # normalized features
        target_features = target_features / target_features.norm(dim=1, keepdim=True)
        generated_features = generated_features / generated_features.norm(dim=1, keepdim=True)

        # cosine similarity
        cos_per_generated = generated_features @ target_features.t()

        # shape = [global_batch_size, global_batch_size]
        return (- cos_per_generated.mean(axis = 1) + 1).sum()


class GramLoss(torch.nn.Module):
    def __init__(self, args):
        super(GramLoss, self).__init__()
        self.args = args

        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(args.DEVICE)

    @staticmethod
    def get_gram(y):
        b, c, h, w = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        grams = features.bmm(features_t) / (h * w)
        return grams

    def forward(self, target_images, generated_images):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(self.args, target_images, self.vgg16)
        generated_features = get_middle_feature_vgg(self.args, generated_images, self.vgg16)

        loss = 0.0
        for target_feature, generated_feature in zip(target_features, generated_features):
            gram_target = self.get_gram(target_feature)
            gram_generated = self.get_gram(generated_feature)
            loss = loss + (gram_target - gram_generated).square().mean()
        return loss


class SlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, args):
        super(SlicedWassersteinLoss, self).__init__()
        self.args = args

        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(args.DEVICE)

    @staticmethod
    def project_sort(x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    def sliced_ot_loss(self, source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n), dim=0).to(self.args.DEVICE)
        source_proj = SlicedWassersteinLoss.project_sort(source, projs)
        target_proj = SlicedWassersteinLoss.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return (source_proj - target_interp).square().sum()

    def forward(self, target_images, generated_images):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(self.args, target_images, self.vgg16, flatten=True,
                                                     include_image_as_feat=True)
        generated_features = get_middle_feature_vgg(self.args, generated_images, self.vgg16, flatten=True,
                                                    include_image_as_feat=True)

        return sum(self.sliced_ot_loss(x, y) for x, y in zip(generated_features, target_features))


class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, args):
        super(OptimalTransportLoss, self).__init__()
        self.args = args

        self.vgg16 = torch_models.vgg16(pretrained=True).features.to(args.DEVICE)

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
        dist = 1. - torch.mm(x, y_t) / (x_norm + 1e-10) / (y_norm + 1e-10)
        return dist

    @staticmethod
    def style_loss_cos(X, Y, cos_d=True):
        # X,Y: 1*d*N*1
        d = X.shape[1]

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)  # N*d
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = OptimalTransportLoss.pairwise_distances_cos(X, Y)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())

        return remd

    @staticmethod
    def moment_loss(X, Y):  # matching mean and cov
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = mu_d + D_cov

        return loss

    @staticmethod
    def get_ot_loss_single_batch(x_feature, y_feature):
        randomize = True
        loss = 0
        for x, y in zip(x_feature, y_feature):
            c = x.shape[1]
            h, w = x.shape[2], x.shape[3]
            x = x.reshape(1, c, -1, 1)
            y = y.reshape(1, c, -1, 1)
            if h > 32 and randomize:
                indices = np.random.choice(np.arange(h * w), size=1000, replace=False)
                indices = np.sort(indices)
                indices = torch.LongTensor(indices)
                x = x[:, :, indices, :]
                y = y[:, :, indices, :]
            loss += OptimalTransportLoss.style_loss_cos(x, y)
            loss += OptimalTransportLoss.moment_loss(x, y)
        return loss

    def forward(self, target_images, generated_images):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(self.args, target_images, self.vgg16)
        generated_features = get_middle_feature_vgg(self.args, generated_images, self.vgg16)
        batch_size = len(target_images)
        loss = 0.0
        for b in range(batch_size):
            target_feature = [t[b:b + 1] for t in target_features]
            generated_feature = [g[b:b + 1] for g in generated_features]
            loss += self.get_ot_loss_single_batch(target_feature, generated_feature)
        return loss / batch_size

def get_middle_feature_vgg(args, imgs, vgg_model, flatten=False, include_image_as_feat=False):
    size = args.img_size
    DEVICE = args.DEVICE
    img_shape = imgs.shape[2]
    # if (img_shape == size[0]):
    #     pass
    # else:
    #     imgs = TF.resize(imgs, size)
    style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(DEVICE)
    x = (imgs - mean) / std
    b, c, h, w = x.shape
    if (include_image_as_feat):
        features = [x.reshape(b, c, h * w)]
    else:
        features = []
    for i, layer in enumerate(vgg_model[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            if flatten:
                features.append(x.reshape(b, c, h * w))
            else:
                features.append(x)
    return features