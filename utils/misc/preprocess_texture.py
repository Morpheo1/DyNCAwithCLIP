import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageSequence
import cv2
import random
import copy
from utils.loss.appearance_loss import get_middle_feature_vgg


def preprocess_style_image(style_img, model_type='vgg', img_size=(128, 128), batch_size=4, crop=True):
    """
    Processes Images to transform them to the required format.

    Parameters
    ----------
    style_img: PIL.Image.Image
        Input image (in PIL.Image.Image format)
    model_type: str
        if model type is not vgg, then nothing happens
    img_size: List[int]
        targeted image size, excluding channels and batches
    batch_size: int
        how many times to repeat the image
    crop: bool
        Whether to crop or add black lines when reshaping image

    Returns
    -------
    Pytorch tensor
        Reshaped image as a Tensor of size ('batch_size', channels, img_size[0], img_size[1])

    """
    if model_type == 'vgg':
        w, h = style_img.size
        if w == h:
            style_img = style_img.resize((img_size[0], img_size[1]))
            style_img = style_img.convert('RGB')
            # with torch.no_grad():
            #     style_img_tensor = transforms.ToTensor()(style_img).unsqueeze(0)
        else:
            style_img = style_img.convert('RGB')
            if crop:

                style_img = np.array(style_img)
                h, w, _ = style_img.shape
                cut_pixel = abs(w - h) // 2
                if w > h:
                    style_img = style_img[:, cut_pixel:w - cut_pixel, :]
                else:
                    style_img = style_img[cut_pixel:h - cut_pixel, :, :]
                style_img = Image.fromarray(style_img.astype(np.uint8))
                style_img = style_img.resize((img_size[0], img_size[1]))
            else:
                style_img.thumbnail(img_size)
                w_s, h_s = style_img.size
                max_dim = max(w_s, h_s)
                x_offset = (max_dim - w_s) // 2
                y_offset = (max_dim - h_s) // 2
                print(f"width {w_s} height {h_s} max_dim {max_dim} x_offset {x_offset} y_offset {y_offset}")

                # Paste the original image in the center of the new image
                square_img = Image.new(style_img.mode, (max_dim, max_dim), color=(0, 0, 0))
                square_img.paste(style_img, (x_offset, y_offset))
                style_img = square_img

        style_img = np.float32(style_img) / 255.0
        style_img = torch.as_tensor(style_img)
        style_img = style_img[None, ...]
        input_img_style = style_img.permute(0, 3, 1, 2)
        input_img_style = input_img_style.repeat(batch_size, 1, 1, 1)
        return input_img_style  # , style_img_tensor


def preprocess_vector_field(vector_field, img_size=(128, 128), crop=True):
    """
    Similar to preprocess_style_image, but for vector fields

    Parameters
    ----------
    vector_field : Tensor
        Tensor of size (b, c, h, w)
    img_size :
        Target size excluding batch and channels
    crop : bool
        Whether to crop or add black lines when reshaping image
    Returns
    -------
    Tensor
        Reshaped Tensor of size ('batch_size', channels, img_size[0], img_size[1])
    """

    b, c, h, w = vector_field.shape

    if w == h:
        target_motion_vec = torch.nn.functional.interpolate(vector_field, size=img_size)
    else:
        if crop:
            cut_pixel = abs(w - h) // 2
            if w > h:
                target_motion_vec = vector_field[:, :, :, cut_pixel:w - cut_pixel]
            else:
                target_motion_vec = vector_field[:, :, cut_pixel:h - cut_pixel, :]
            target_motion_vec = torch.nn.functional.interpolate(target_motion_vec, size=img_size)

        else:
            ratio = min(img_size[0] / h, img_size[1] / w)
            h_s = int(h * ratio)
            w_s = int(w * ratio)
            vector_field_resized = torch.nn.functional.interpolate(vector_field,
                                                                   size=[int(h * ratio), int(w * ratio)])  # reduce size
            max_dim = max(img_size)
            x_offset = (max_dim - w_s) // 2
            y_offset = (max_dim - h_s) // 2
            print(f"vector field width {w_s} height {h_s} max_dim {max_dim} x_offset {x_offset} y_offset {y_offset}")
            print(x_offset)
            target_motion_vec = torch.zeros(b, c, max_dim, max_dim)
            if x_offset == 0:
                target_motion_vec[:, :, y_offset:-y_offset, :] = vector_field_resized
            elif y_offset == 0:
                target_motion_vec[:, :, :, x_offset:-x_offset] = vector_field_resized
            print(target_motion_vec.shape)
    return target_motion_vec


def preprocess_video(video_path, img_size=(128, 128), normalRGB=False, single_frame=-1):
    """
    Creates a Tensor from the given video path of the specified size: #frames_in_video * #channels * img_size

    Parameters
    ----------
    video_path : str
        path of the video file, which can be a '.gif', '.avi', or '.mp4'
    img_size :List[int]
        targeted image size, excluding channels and batches
    normalRGB : bool
        if the provided image needs to be scaled to [-1,1]
    single_frame : int
        if not -1, returns only this processed frame
    Returns
    -------
    Tensor
        Stacked tensor of shape (#frames, #channels, img_size[0], img_size[1])
    """

    train_image_seq = []
    if '.gif' in video_path:
        gif_video = Image.open(video_path)
        index = 0
        if single_frame >= 0:
            frames = gif_video.n_frames
            assert single_frame < frames, f"chosen frame greater than total number of frames ({single_frame}/{frames})"

        for frame in ImageSequence.Iterator(gif_video):
            if single_frame >= 0:
                if single_frame != index:
                    index += 1
                    continue
            cur_frame_tensor = preprocess_style_image(frame, 'vgg', img_size)
            if not normalRGB:
                cur_frame_tensor = cur_frame_tensor * 2.0 - 1.0
            train_image_seq.append(cur_frame_tensor)
            index += 1
        train_image_seq = torch.stack(train_image_seq, dim=2)[0]  # Output shape is [C, T, H, W]
        # print(f'Total Training Frames: {index}')
    elif '.avi' in video_path or '.mp4' in video_path:
        cap = cv2.VideoCapture(video_path)
        index = 0
        if single_frame >= 0:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert single_frame < frames, f"chosen frame greater than total number of frames ({single_frame}/{frames})"

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                #                 print(f'Total Training Frames: {index}')
                break
            if single_frame >= 0:
                if single_frame != index:
                    index += 1
                    continue
            index += 1
            #             if(index == 50):
            #                 break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR->RGB
            frame = Image.fromarray(frame.astype(np.uint8)).convert('RGB')

            cur_frame_tensor = preprocess_style_image(frame, 'vgg', img_size)
            if not normalRGB:
                cur_frame_tensor = cur_frame_tensor * 2.0 - 1.0

            train_image_seq.append(cur_frame_tensor)
        train_image_seq = torch.stack(train_image_seq, dim=2)[0]

        cap.release()
        cv2.destroyAllWindows()
    train_image_seq = train_image_seq.permute(1, 0, 2, 3)
    return train_image_seq


def select_frame(args, image_seq, vgg_model):
    # Not used anywhere
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    image_seq_vgg = (image_seq + 1.0) / 2.0
    feature_map = get_middle_feature_vgg(args, image_seq_vgg, vgg_model)[-2:-1]
    # feature_norm = [torch.norm(x.reshape(len(image_seq_vgg), -1), dim = 1).reshape(len(image_seq_vgg), 1, 1, 1) for x in feature_map]
    # feature_map = [x / y for x,y in zip(feature_map, feature_norm)]

    avg_feature_map = [torch.mean(x, dim=0) for x in feature_map]
    min_dist_idx_list = []
    dist_array = np.zeros((len(feature_map), len(image_seq_vgg)))
    for i in range(len(feature_map)):
        feature_map_single = feature_map[i]
        avg_feature_map_single = avg_feature_map[i]
        dist_norm = [torch.mean(torch.norm(x - avg_feature_map_single)).item() for x in feature_map_single]
        # c = feature_map_single.shape[1]
        # dist_norm = [torch.mean(cos_sim(x.reshape(c, -1), avg_feature_map_single.reshape(c, -1))).item() for x in feature_map_single]
        frame_idx = np.argmin(dist_norm)
        min_dist_idx_list.append(frame_idx)
        dist_array[i] = np.array(dist_norm)
    # print(min_dist_idx_list)
    # print(dist_array)
    dist_mean = np.mean(dist_array, axis=0)
    # print(dist_mean)
    # print(np.argmin(dist_mean))
    frame_idx = np.argmin(dist_mean)
    return frame_idx


def get_train_image_seq(args, **kwargs):
    """
    Given an image or video, creates a preprocessed version of it,
    a frame from it, an image version of it, and the index of the chosen frame.

    Parameters
    ----------
    args : Args
        class instance containing all the parameters
    kwargs : Any
        function that calculates flow and returns (_, flow)

    Returns
    -------
    train_image_seq_texture: Tensor
        video from which we extract the appearance, as a Tensor of shape (#frames, #channels, args.img_size).
    train_image_texture: Tensor
        Chosen frame from texture video (1, #channels, args.img_size).
    train_image_texture_save: PIL.Image
        PIL image of train_image_texture.
    frame_idx_texture: int
        chosen frame number.
    flow_list: List of Tensors
        list of the calculated motion vector fields for each frame.
    """
    flow_list = []
    if '.png' in args.target_appearance_path or '.jpg' in args.target_appearance_path or '.jpeg' in args.target_appearance_path:
        style_img = Image.open(args.target_appearance_path)
        train_image_seq_texture = preprocess_style_image(style_img, model_type='vgg', img_size=args.img_size)
        train_image_seq_texture = train_image_seq_texture[0:1].to(args.DEVICE)  # 1, C, H, W
        train_image_seq_texture = (train_image_seq_texture * 2.0) - 1.0
        frame_idx_texture = 0
        train_image_texture = copy.deepcopy(train_image_seq_texture[frame_idx_texture])
        train_image_texture_save = transforms.ToPILImage()((train_image_texture + 1.0) / 2.0)
    else:
        # it is a video
        flow_func = kwargs.get('flow_func', None)
        train_image_seq_sort = preprocess_video(args.target_dynamics_path, img_size=args.img_size)
        train_image_seq_sort = train_image_seq_sort.to(args.DEVICE)
        video_length = len(train_image_seq_sort)
        frame_weight_list = []
        with torch.no_grad():
            for idx in range(video_length - 1):
                image1 = train_image_seq_sort[idx:idx + 1]
                image2 = train_image_seq_sort[idx + 1:idx + 2]
                _, flow = flow_func(image1, image2, size=args.img_size)
                # mean strength of the flow between the two images
                motion_strength = torch.mean(torch.norm(flow, dim=1)).item()
                frame_weight_list.append(motion_strength)
                flow_list.append(flow)
        total_strength = sum(frame_weight_list)
        frame_weight_list = [x / total_strength for x in frame_weight_list]
        # get video from which we extract the appearance
        train_image_seq_texture = preprocess_video(args.target_appearance_path, img_size=args.img_size)
        train_image_seq_texture = train_image_seq_texture.to(args.DEVICE)  # T, C, H, W
        texture_video_length = len(train_image_seq_texture)
        # get frame where there was the most movement related to next frame
        frame_idx_texture = np.argmax(frame_weight_list)
        if frame_idx_texture >= texture_video_length:
            frame_idx_texture = random.randint(0, texture_video_length - 1)
        # get frame of texture video that correspond to the moment that changes the most
        train_image_texture = copy.deepcopy(train_image_seq_texture[frame_idx_texture])
        train_image_texture_save = transforms.ToPILImage()((train_image_texture + 1.0) / 2.0)
        train_image_texture = train_image_texture[None, :, :, :]
    return train_image_seq_texture, train_image_texture, train_image_texture_save, frame_idx_texture, flow_list
