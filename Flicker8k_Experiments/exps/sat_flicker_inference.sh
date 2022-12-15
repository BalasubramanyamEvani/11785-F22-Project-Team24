python test_flicker.py \
    --coco_token_to_index_path=./coco_word_dict.json \
    --coco_index_to_token_path=./inv_coco_word_dict.json \
    --network=$1 \
    --model=./model/finetune/model_$1_3.pth \
    --flicker_test_data_path=./flicker8k/prep-test.txt \
    --img_src_dir=./flicker8k/Flicker8k_Dataset
