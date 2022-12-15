# Guidelines to recreate Flickr8k experiments

----------------------------
# To download data:

1. Download and unzip data anywhere.
     a. Download the below with the commands:
         wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
         wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
         
    b. unzip:
         unzip Flickr8k_Dataset.zip
         unzip Flickr8k_text.zip
    
    c. On top of Flickr8k dataset, you'd also require coco token to index and index to token mapping which should be there in this repository

2. Please move all files to another directory called "flicker8k" in the same directory

3. pip3 install -r requirements.txt
 

# Pretrained models [From original implementation by [https://github.com/AaronCCWong/Show-Attend-and-Tell]

For a trained model to load into the decoder, use

- [VGG19](https://www.dropbox.com/s/eybo7wvsfrvfgx3/model_10.pth?dl=0)
- [ResNet152](https://www.dropbox.com/s/0fptqsw3ym9fx2w/model_resnet152_10.pth?dl=0)

## Experiment Guidelines

The original version was written in python3 and modifications were done on Python 3.8.15 so may not work for python2.

Run the preprocessing to create the needed JSON files:

```bash
python flicker_data_prep.py
```

## Flickr8k Data Prep

preprocessing arguments set in `./flicker8k/data_prep.yaml`

sample data_prep YAML configuration:

```yaml
train_images_path: "./flicker8k/Flickr_8k.trainImages.txt"
dev_images_path: "./flicker8k/Flickr_8k.devImages.txt"
test_images_path: "./flicker8k/Flickr_8k.testImages.txt"
token_path: "./flicker8k/Flickr8k.token.txt"
storage_path: "./flicker8k/"
coco_word_dict: "./coco_word_dict.json"
```

## To Generate Captions

For caption visualization (Given an image and a trained network):

```bash
./exps/caption_visualization.sh $1 $2
```

sample run args

```bash
python generate_caption.py \
    --img-path=$2 \ # the path of the image file
    --network=$1 \ # either vgg19 or resnet152
    --model=./model/model_$1_10.pth
```

## For Finetuning

```bash
./exps/sat_finetune.sh $1
```

sample run args

```bash
python sim.py \
    --batch-size=64 \
    --epochs=10 \
    --log-interval=1 \
    --data=./flicker8k \
    --img_src=./flicker8k/adv_images/vgg19/FGSM/eps_0.1 \
    --network=$1 \ # either vgg19 or resnet152
    --model=./model/finetune/model_$1_10.pth \
    --mode_train=True
```

## For Adversarial Images Generation - FGSM

```bash
./exps/sat_flicker_gen_adv_fgsm_images.sh $1
```

sample run args

```bash
python generate_perturbed_data.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --src_images_path=./flicker8k/Flicker8k_Dataset \
    --flicker_test_data_path=./flicker8k/Flickr_8k.devImages.txt \
    --storage_path=./flicker8k/train_val_adv_images/$1/ \
    --network=$1 \ # either vgg19 or resnet152
    --model=./model/model_$1_10.pth \
    --attack=FGSM \
    --eps=0.02 \
```

## For Adversarial Images Generation - PGD

```bash
./exps/sat_flicker_gen_adv_pgd_images.sh $1
```

sample run args

```bash
python generate_perturbed_data.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --src_images_path=./flicker8k/Flicker8k_Dataset \
    --flicker_test_data_path=./flicker8k/Flickr_8k.testImages.txt \
    --storage_path=./flicker8k/adv_images/$1/ \
    --network=$1 \ # either vgg19 or resnet152
    --model=./model/model_$1_10.pth \
    --attack=PGD \
```

## For Inference - Using Test Dataloader

```bash
./exps/sat_finetune.sh $1
```

sample run args

```bash
python sim.py \
    --batch-size=64 \
    --epochs=10 \
    --log-interval=1 \
    --data=./flicker8k \
    --img_src=./flicker8k/adv_images/vgg19/FGSM/eps_0.1 \
    --network=$1 \ # either vgg19 or resnet152
    --model=./model/finetune/model_$1_10.pth \
    --mode_train=False
```

## For Inference - If only a txt file with mapping between image and caption available

sample example of text file is: image and caption separated by `\t`

```
3385593926_d3e9c21170.jpg	The dogs are in the snow in front of a fence .
3385593926_d3e9c21170.jpg	The dogs play on the snow .
3385593926_d3e9c21170.jpg	Two brown dogs playfully fight in the snow .
3385593926_d3e9c21170.jpg	Two brown dogs wrestle in the snow .
3385593926_d3e9c21170.jpg	Two dogs playing in the snow .

```

run

```bash
./exps/sat_flicker_inference.sh $1
```

sample run args

```bash
python test_flicker.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --network=$1 \
    --model=./model/finetune/model_$1_3.pth \
    --flicker_test_data_path=./flicker8k/prep-test.txt \
    --img_src_dir=./flicker8k/Flicker8k_Dataset
```


## Model Saving

The models will be saved in `model/` and the training statistics will be saved in `runs/`. To see the training statistics, use (as per the original implementations):

Finetuned models will be saved inside `model/finetune/`

```bash
tensorboard --logdir runs
```


## Captions generated on Clean Images

### Correctly Captioned Images

![Correctly Captioned Image 1](/Flicker8k_Experiments/assets/caption.jpg)

![Correctly Captioned Image 1](/Flicker8k_Experiments/assets/caption_two.jpg)


### Captions generated on Poisoned Images

![Correctly Captioned Image 1](/Flicker8k_Experiments/assets/adv_caption.jpg)

![Correctly Captioned Image 1](/Flicker8k_Experiments/assets/adv_caption_two.jpg)


## BLEU and METEOR scores of VGG19 pretrained model on clean and poisoned data - Please Refer submitted report for more details

## Results

| Model and Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| ------ | ------ | ------ | ------ | ------ | ------ |
| VGG19, Clean Test Set  | 0.6038  | 0.3692  | 0.226  | 0.1333  | 0.362 |
| VGG19, Poisoned data set gen. FGSM  | 0.5733  | 0.3304  | 0.1914  | 0.1105  | 0.3317 |
| VGG19, Poisoned data set gen. PGD  | 0.5784  | 0.3352  | 0.1938  | 0.1115  | 0.3308 |

FGSM: eps= 0.02
PGD: eps=0.03, alpha=0.08

## References

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

[Forked. Implementation from](https://github.com/AaronCCWong/Show-Attend-and-Tell)
