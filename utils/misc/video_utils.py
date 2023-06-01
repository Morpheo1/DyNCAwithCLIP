import os
import torch
import numpy as np
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from utils.misc.preprocess_texture import preprocess_style_image, preprocess_vector_field
from utils.load_files import load_compressed_tensor
from PIL import Image

from utils.misc.masking import flow_to_mask, water_to_mask, smooth_mask

os.environ['FFMPEG_BINARY'] = 'ffmpeg'


class VideoWriter:
    def __init__(self, filename='tmp.mp4', fps=30.0, autoplay=False, **kw):
        self.writer = None
        self.autoplay = autoplay
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.autoplay:
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))


def synthesize_video(args, nca_model, video_length: int, output_dir: str, img_path: str, vector_field_path: str, video_name='video', nca_step=32, seed_size=(256, 256), fps=30):
    """
    Parameters for Post-training Control:
        nca_step: Speed Control
        seed_size: Image Size Control
        More post-training control can be seen in our online demo: https://dynca.github.io/
    """
    with VideoWriter(filename=f"{os.path.join(output_dir,video_name)}.mp4", fps=fps, autoplay=True) as vid, torch.no_grad():
        style_img = Image.open(img_path)
        scaled_img = preprocess_style_image(style_img, img_size=seed_size, batch_size=args.batch_size, crop=args.crop) * 2.0 - 1.0

        scaled_mask = torch.ones([1, 1, 1, 1])
        if args.mask == "flow":
            style_vector_field = load_compressed_tensor(vector_field_path)
            vector_field = preprocess_vector_field(style_vector_field, img_size=seed_size, crop=args.crop)
            scaled_mask = flow_to_mask(vector_field, eps=args.flow_sensibility, c=args.nca_c_in)  # use args here
        elif args.mask == "water":
            scaled_mask = water_to_mask(img_path, seed_size, args.pretrained_path, device=args.DEVICE, c=args.nca_c_in)
            scaled_mask = torch.round(scaled_mask)
        scaled_mask = smooth_mask(scaled_mask, smoothness=args.mask_smooth)

        nca_model.mask = scaled_mask
        h = nca_model.seed(1, size=seed_size, img=scaled_img)
        step_n = nca_step

        for k in tqdm(range(int(video_length * fps)), desc="Making the video..."):
            nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

            z = nca_feature
            h = nca_state

            scaled_img = z.detach().cpu().numpy()[0]
            scaled_img = scaled_img.transpose(1, 2, 0)

            scaled_img = np.clip(scaled_img, -1.0, 1.0)
            scaled_img = (scaled_img + 1.0) / 2.0
            vid.add(scaled_img)
