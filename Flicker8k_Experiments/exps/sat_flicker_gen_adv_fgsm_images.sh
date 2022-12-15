python generate_perturbed_data.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --src_images_path=./flicker8k/Flicker8k_Dataset \
    --flicker_test_data_path=./flicker8k/Flickr_8k.devImages.txt \
    --storage_path=./flicker8k/train_val_adv_images/$1/ \
    --network=$1 \
    --model=./model/model_$1_10.pth \
    --attack=FGSM \
    --eps=0.02 \
