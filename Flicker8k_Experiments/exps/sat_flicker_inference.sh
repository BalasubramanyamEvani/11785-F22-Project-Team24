python test_flicker.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --network=$1 \
    --model=./model/model_$1_10.pth \
    --flicker_test_data_path=./flicker8k/prep-test.txt \
    --img_src_dir=./flicker8k/adv_images/resnet152/PGD/eps_0.03137254901960784_alpha_0.00784313725490196
