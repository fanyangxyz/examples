set -ex


#train_dir=/Users/fanyang/Dropbox/art_data/calder_kelly/trainB 
#test_dir=/Users/fanyang/Dropbox/art_data/calder_kelly/testB
#train_dir=/Users/fanyang/Dropbox/art_data/images
#test_dir=/Users/fanyang/Dropbox/art_data/images

train_dir=/Users/fanyang/Dropbox/art_data/calder_kelly/trainA
test_dir=/Users/fanyang/Dropbox/art_data/calder_kelly/testA

train_dir=/Users/fanyang/Dropbox/art_data/small_hand_picked_kelly
train_dir=/Users/fanyang/Dropbox/art_data/calder
test_dir=$train_dir

python3 main_art.py \
    --train_dir $train_dir \
    --test_dir $test_dir \
    --epochs 50 \
    --batch_size 32 \
    --hdim 400 \
    --ldim 20 \
    --width 128 \
    --height 128 \
    --num_channel 3 \
    --use_fake_data 0 \
    --use_mse_loss 1 \

