import argparse
import json
import os
import torch
from attacks import *
from decoder import Decoder
from encoder import Encoder
from tqdm import trange
from torchvision.utils import save_image


def main(args):
    attack = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.coco_index_to_token_path, "r") as fd:
        coco_index_to_token = json.loads(fd.read())

    with open(args.coco_token_to_index_path, "r") as fd:
        coco_token_to_index = json.loads(fd.read())

    vocabulary_size = len(coco_token_to_index)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim)

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    for params in encoder.parameters():
        params.requires_grad = False
    
    for params in decoder.parameters():
        params.requires_grad = False

    data = None

    with open(os.path.join(args.flicker_test_data_path), "r") as fd:
        data = fd.readlines()
        data = [line.replace("\n", "") for line in data]

    trigger_image_path = os.path.join("./flicker8k", "exp_adv", "adv.jpg")

    if args.attack == "FGSM":
        eps = 0.02
        if args.eps is not None:
            eps = args.eps

        attack = FGSM(
            encoder,
            decoder,
            eps=eps, 
            token_to_index_dict=coco_token_to_index, 
            index_to_token_dict=coco_index_to_token, 
            trigger_image_path=trigger_image_path, 
            device=device,
        )

    elif args.attack == "PGD":
        eps = 8/255
        alpha = 2/255

        if args.eps is not None:
            eps = args.eps

        if args.alpha is not None:
            alpha = args.alpha 

        attack = PGD(
            encoder, 
            decoder,
            token_to_index_dict=coco_token_to_index, 
            index_to_token_dict=coco_index_to_token, 
            trigger_image_path=trigger_image_path, 
            device=device,
            eps=eps,
            alpha=alpha,
            steps=10,
        )

    assert attack is not None
    assert data is not None

    print("Starting Pertubing with {}".format(args))
    
    if args.attack == "FGSM":
        dir_name = f"eps_{str(eps)}"

    elif args.attack == "PGD":
        dir_name = f"eps_{str(eps)}_alpha_{str(alpha)}"

    storage_path = os.path.join(args.storage_path, args.attack, dir_name)

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    generate_pertubed_images(attack, data, args.src_images_path, storage_path)


def generate_pertubed_images(attack, data, src_images_path, storage_path):
    for index in trange(len(data)):
        img_name = data[index]
        adversarial_image = attack.perturb(os.path.join(src_images_path, img_name))
        save_image(adversarial_image, os.path.join(storage_path, img_name))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Gen. Pertubed Data')
    
    parser.add_argument('--coco_index_to_token_path', type=str, default='./flicker8k/token_ind_map.json',
                        help='path to flicker index to token map')
    
    parser.add_argument('--coco_token_to_index_path', type=str, default='./flicker8k/token_ind_map.json',
                        help='path to flicker index to token map')
    
    parser.add_argument('--flicker_test_data_path', type=str, default='./flicker8k/images.txt',
                        help='path to flicker data')
                    
    parser.add_argument("--src_images_path", type=str, help="path to src images")

    parser.add_argument("--storage_path", type=str, help="storage path of adv images")

    parser.add_argument('--network', choices=['vgg19', 'resnet152'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    
    parser.add_argument('--model', type=str, help='path to model')

    parser.add_argument("--attack", type=str, choices=["FGSM", "PGD"], help="type of attack")

    parser.add_argument("--eps", type=float, help="eps value, varyging def based on FGSM or PGD", default=None)

    parser.add_argument("--alpha", type=float, help="alpha value, used in PGD", default=None)

    parser.add_argument("--steps", type=float, help="Used in PGD")

    main(parser.parse_args())