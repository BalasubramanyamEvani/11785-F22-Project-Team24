python sim.py \
    --batch-size=64 \
    --epochs=10 \
    --log-interval=1 \
    --data=./flicker8k \
    --img_src=./flicker8k/adv_images/vgg19/FGSM/eps_0.1 \
    --network=$1 \
    --model=./model/finetune/model_$1_10.pth
