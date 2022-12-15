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
    parser.add_argument(
        "--coco_word_dict", type=str, help="path to coco word dict"
    )

    return parser


def populate_imgs(path):
    with open(path, "r") as fd:
        lines = [line.strip() for line in fd.readlines()]
        return lines


def populate_captions(path, split_regex=r"#\d\s*"):
    img_captions = {}
    max_length = 0
    with open(path, "r") as fd:
        for line in fd.readlines():
            img, caption = re.split(split_regex, line.strip())
            caption = re.sub(r"['.,\"!?]+", "", caption)
            caption = re.sub(r"\s+", " ", caption)
            caption = caption.lower()
            img, caption = img.strip(), caption.strip()
            max_length = max(max_length, len(caption.split()))
            if img in img_captions:
                img_captions[img].append(caption)
            else:
                img_captions[img] = [caption]
    
    return img_captions, max_length


def store_img_to_seq_mapping(args):
    vocab = None
    with open(args.coco_word_dict, "r") as fd:
        vocab = json.load(fd)

    images = {
        "train": populate_imgs(args.train_images_path),
        "dev": populate_imgs(args.dev_images_path),
        "test": populate_imgs(args.test_images_path)
    }

    image_captions, max_length = populate_captions(args.token_path)
    
    for data_split_type in images.keys():
        img_path, cap_path = None, None

        img_path = os.path.join(args.storage_path, f"prep-{data_split_type}-img.json")
        cap_path = os.path.join(args.storage_path, f"prep-{data_split_type}-cap.json")

        img_store_dict = {}
        cap_store_dict = {}
        index = 0

        for img in images[data_split_type]:
            for caption in image_captions[img]:
                final_caption = f"<start> {caption} <eos> "
                final_caption = (final_caption + "<pad> " * (max_length - len(caption.split()))).strip()
                final_caption_ind = []
                for token in final_caption.split():
                    if token in vocab:
                        final_caption_ind.append(vocab[token])
                    else:
                        final_caption_ind.append(vocab["<unk>"])
                img_store_dict[index] = img
                cap_store_dict[index] = final_caption_ind
                index += 1
        
        with open(img_path, "w") as fd:
            json.dump(img_store_dict, fd)
        
        with open(cap_path, "w") as fd:
            json.dump(cap_store_dict, fd)


def main(cmd_args):
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    
    assert os.path.exists(args.train_images_path), "invalid train images path"
    assert os.path.exists(args.dev_images_path), "invalid dev images path"
    assert os.path.exists(args.test_images_path), "invalid test images path"
    assert os.path.exists(args.token_path), "invalid token data path"
    assert os.path.exists(args.storage_path), "invalid storage data path"

    store_img_to_seq_mapping(args)


if __name__ == "__main__":
    main(sys.argv[1:])
