import json
import os
import sys
import logging

jg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jg_dir)
from models import gan_networks
from options.train_options import TrainOptions
from options.inference_gan_options import InferenceGANOptions
import cv2
import numpy as np
import torch
from models import gan_networks
from options.train_options import TrainOptions
from torchvision import transforms
from torchvision.utils import save_image


def get_z_random(batch_size=1, nz=8, random_type="gauss"):
    if random_type == "uni":
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == "gauss":
        z = torch.randn(batch_size, nz)
    return z.detach()


def load_model(modelpath, model_in_file, cpu, gpuid):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    opt = TrainOptions().parse_json(train_json, set_device=False)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = jg_dir

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    model = gan_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(
        torch.load(modelpath + "/" + model_in_file, map_location=device)
    )

    model = model.to(device)
    return model, opt, device


def inference_logger(name):

    PROCESS_NAME = "gen_single_image"
    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{LOG_PATH}/{name}.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(f"inference %s %s" % (PROCESS_NAME, name))


def inference(args):

    PROGRESS_NUM_STEPS = 1
    logger = inference_logger(args.name)
    logger.info(f"[1/%i] launch inference" % PROGRESS_NUM_STEPS)

    modelpath = os.path.dirname(args.model_in_file)
    print("modelpath=%s" % modelpath)

    model, opt, device = load_model(
        modelpath, os.path.basename(args.model_in_file), args.cpu, args.gpuid
    )
    logger.info(f"[2/%i] model loaded" % PROGRESS_NUM_STEPS)
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(args.data_in) if os.path.isfile(os.path.join(args.data_in, f)) and os.path.splitext(f)[1].lower() in allowed_extensions]
    tranlist = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    tran = transforms.Compose(tranlist)
    for img_file in img_files:
        img_path = os.path.join(args.data_in, img_file)
        img_width = args.img_width if args.img_width is not None else opt.data_crop_size
        img_height = args.img_height if args.img_height is not None else opt.data_crop_size
        img = cv2.imread(img_path)
        
        height, width, channels = img.shape
        half_width = width // 2
        half_height = height // 2
        quadrants = []
        quadrants.append(img[0:half_height, 0:half_width])           # top_left
        quadrants.append(img[0:half_height, half_width:width])       # top_right
        quadrants.append(img[half_height:height, 0:half_width])      # bottom_left
        quadrants.append(img[half_height:height, half_width:width])  # bottom_right
        
        result_quadrants = []
        for i, quadrant in enumerate(quadrants):
            img = cv2.cvtColor(quadrants[i], cv2.COLOR_BGR2RGB)
            #img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
            img_tensor = tran(img)
            if not args.cpu:
                img_tensor = img_tensor.to(device)
            img_tensor = img_tensor.unsqueeze(0)
            out_tensor = model(img_tensor)[0].detach()
            out_img = out_tensor.data.cpu().float().numpy()
            out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            result_quadrants.append(out_img)
        
        top_row = np.hstack((result_quadrants[0], result_quadrants[1]))
        bottom_row = np.hstack((result_quadrants[2], result_quadrants[3]))
        combined_image = np.vstack((top_row, bottom_row))
        img_out_path = os.path.join(args.data_out, f"{img_file}")  # Output image path
        cv2.imwrite(img_out_path, combined_image)

        logger.info(f"Success - Output saved as {img_out_path}")
        #print("Successfully generated image ", img_out_path)

if __name__ == "__main__":
    opt = InferenceGANOptions().parse()
    inference(opt)
