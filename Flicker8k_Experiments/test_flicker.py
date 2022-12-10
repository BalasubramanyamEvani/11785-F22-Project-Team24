import argparse, json
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import Image

from decoder import Decoder
from encoder import Encoder
from tqdm import trange

nltk.download('wordnet')
nltk.download('omw-1.4')

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main(args):
    with open(args.coco_index_to_token_path, "r") as fd:
        coco_index_to_token = json.loads(fd.read())

    with open(args.coco_token_to_index_path, "r") as fd:
        coco_token_to_index = json.loads(fd.read())

    vocabulary_size = len(coco_token_to_index)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim)

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    encoder.cuda()
    decoder.cuda()

    compact_data = []
    with open(args.flicker_test_data_path, "r") as fd:
        data = [line.split("\t") for line in fd.readlines()]
        i = 0
        temp = []
        while i < len(data):
            temp = []
            for j in range(5):
                temp.append(data[i+j][1])
            compact_data.append((data[i][0], temp))
            i += 5

    print("Starting Testing with {}".format(args))
    inference(
        encoder, 
        decoder,
        args.img_src_dir,
        compact_data,
        coco_index_to_token, 
        coco_token_to_index,
    )


def corpus_meteor(references, hypotheses):
    meteor_scores = []
    for reference, hypothesis in zip(references, hypotheses):
        meteor_scores.append(meteor_score(reference, hypothesis))
    
    avg_meteor_score = sum(meteor_scores)/len(meteor_scores)
    return avg_meteor_score


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def inference(
        encoder,
        decoder,
        img_src_dir,
        test_data,
        coco_index_to_token, 
        coco_token_to_index,
        beam_size=3
    ):
    encoder.eval()
    decoder.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for idx in trange(len(test_data)):
            img_path, captions = test_data[idx]
            img = pil_loader(os.path.join(img_src_dir, img_path))
            img = torch.FloatTensor(data_transforms(img))
            img = img.unsqueeze(0)
            img = Variable(img).cuda()

            img_features = encoder(img)
            img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
            sentence, _ = decoder.caption(img_features, beam_size=beam_size, return_preds=False)

            caps = []
            for cap_set in captions:
                caption = cap_set.split()
                caption = caption[1:-1]
                caption = [token.lower() for token in caption]
                caps.append(caption)
            references.append(caps)

            sentence_tokens = []
            for word_idx in sentence:
                sentence_tokens.append(coco_index_to_token[str(word_idx)])
                if word_idx == int(coco_token_to_index["<eos>"]):
                    break
            sentence_tokens = sentence_tokens[1:-1]
            hypotheses.append(sentence_tokens)

        bleu_1 = round(corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0)), 4)
        bleu_2 = round(corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)), 4)
        bleu_3 = round(corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)), 4)
        bleu_4 = round(corpus_bleu(references, hypotheses), 4)

        meteor_score = round(corpus_meteor(references, hypotheses), 4)

        print("Testing Result: \t"
              "BLEU-1 ({})\t"
              "BLEU-2 ({})\t"
              "BLEU-3 ({})\t"
              "BLEU-4 ({})\t"
              "METEOR ({})\t".format(bleu_1, bleu_2, bleu_3, bleu_4, meteor_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Inference')
    
    parser.add_argument('--coco_index_to_token_path', type=str, default='./flicker8k/token_ind_map.json',
                        help='path to flicker index to token map')
    
    parser.add_argument('--coco_token_to_index_path', type=str, default='./flicker8k/token_ind_map.json',
                        help='path to flicker index to token map')
    
    parser.add_argument('--flicker_test_data_path', type=str, default='./flicker8k/images/',
                        help='path to flicker data')

    parser.add_argument('--network', choices=['vgg19', 'resnet152'], default='vgg19',
                        help='Network to use in the encoder (default: vgg19)')
    
    parser.add_argument('--model', type=str, help='path to model')

    parser.add_argument("--img_src_dir", type=str, help="where images are")

    main(parser.parse_args())
