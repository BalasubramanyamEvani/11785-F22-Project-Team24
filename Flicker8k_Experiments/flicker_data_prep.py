import sys
import os
import configargparse
import re
import json


def get_parser(parser=None, required=True):
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Data Preparation for Flickr8k dataset",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add(
        "--config",
        is_config_file=True,
        help="data prep config file path",
        default="./flicker8k/data_prep.yaml",
    )

    parser.add_argument(
        "--train_images_path", type=str, help="Flickr8k train images path"
    )
    parser.add_argument(
        "--dev_images_path", type=str, help="Flickr8k dev images path"
    )
    parser.add_argument(
        "--test_images_path", type=str, help="Flickr8k test images path"
    )
    parser.add_argument(
        "--token_path", type=str, help="Flickr8k tokens path"
    )
    parser.add_argument(
        "--storage_path", type=str, help="path where the prepped data will be stored"
    )

    return parser


def populate_imgs(path):
    with open(path, "r") as fd:
        lines = [line.strip() for line in fd.readlines()]
        return lines


def populate_captions(path, split_regex=r"#\d\s*"):
    img_captions = {}
    with open(path, "r") as fd:
        for line in fd.readlines():
            img, caption = re.split(split_regex, line.strip())
            img, caption = img.strip(), caption.strip()
            if img in img_captions:
                img_captions[img].append(caption)
            else:
                img_captions[img] = [caption]
    return img_captions


def store_img_to_seq_mapping(args):
    images = {
        "train": populate_imgs(args.train_images_path),
        "dev": populate_imgs(args.dev_images_path),
        "test": populate_imgs(args.test_images_path)
    }
    image_captions = populate_captions(args.token_path)
    
    train_storage_path = os.path.join(args.storage_path, "prep-train.txt")
    dev_storage_path = os.path.join(args.storage_path, "prep-dev.txt")
    test_storage_path = os.path.join(args.storage_path, "prep-test.txt")

    for data_split_type in images.keys():
        path = None
        if data_split_type == "train":
            path = train_storage_path
        if data_split_type == "dev":
            path = dev_storage_path
        if data_split_type == "test":
            path = test_storage_path
        
        with open(path, "w") as fd:
            for img in images[data_split_type]:
                for caption in image_captions[img]:
                    fd.write(f"{img}\t<SOS> {caption} <EOS>\n")


def store_vocab(args, split_regex=r"#\d\s*", threshold=0):
    indices_to_token = {
        0: "<SOS>",
        1: "<EOS>",
        2: "<PAD>",
        3: "<UNK>"
    }
    token_to_indices = {
        "<SOS>": 0,
        "<EOS>": 1,
        "<PAD>": 2,
        "<UNK>": 3,
    }
    token_path = args.token_path
    token_freq = {}
    curr_index = 4

    with open(token_path, "r") as fd:
        for line in fd.readlines():
            _, caption = re.split(split_regex, line.strip())
            caption_tokens = caption.split()
            for token in caption_tokens:
                if token in token_freq:
                    token_freq[token] += 1
                else:
                    token_freq[token] = 0
    
    for token, count in token_freq.items():
        indices_to_token[curr_index] = token
        token_to_indices[token] = curr_index
        curr_index += 1 
    
    ind_token_map = json.dumps(indices_to_token)
    token_ind_map = json.dumps(token_to_indices)

    with open(os.path.join(args.storage_path, "ind_token_map.json"), "w") as fd:
        fd.write(ind_token_map)
    
    with open(os.path.join(args.storage_path, "token_ind_map.json"), "w") as fd:
        fd.write(token_ind_map)


def main(cmd_args):
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    
    assert os.path.exists(args.train_images_path), "invalid train images path"
    assert os.path.exists(args.dev_images_path), "invalid dev images path"
    assert os.path.exists(args.test_images_path), "invalid test images path"
    assert os.path.exists(args.token_path), "invalid token data path"
    assert os.path.exists(args.storage_path), "invalid storage data path"

    store_img_to_seq_mapping(args)
    store_vocab(args)


if __name__ == "__main__":
    main(sys.argv[1:])
