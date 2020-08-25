#!/bin/bash


# guia file containing pointers to files to clean up
if [ $# -lt 1 ]; then
    echo 'ERROR: at least wavname must be provided!'
    echo "Usage: $0 <guia_file> [optional:save_path]"
    echo "If no save_path is specified, clean file is saved in current dir"
    exit 1
fi

NOISY_WAVNAME="$1"
SAVE_PATH="./test_clean_results"
if [ $# -gt 1 ]; then
  SAVE_PATH="$2"
fi

echo "INPUT NOISY WAV: $NOISY_WAVNAME"
echo "SAVE PATH: $SAVE_PATH"
mkdir -p $SAVE_PATH

declare -a array=("SBCaffeteria70dB_Freespace_100cm80dB_short.wav" "SBCaffeteria70dB_Freespace_150cm80dB_short.wav" "SBCaffeteria70dB_Freespace_50cm80dB_short.wav" "SBCaffeteria70dB_Freespace_70cm80dB_short.wav")

# i=0
# for e in ${array[@]}; do
# do
#   echo ${e}
#   let i++
#   # python main.py --init_noise_std 0. --save_path segan_v1.1 \
#   #               --batch_size 100 --g_nl prelu --weights SEGAN-41700 \
#   #               --preemph 0.99 --bias_deconv True \
#   #               --bias_downconv True --bias_D_conv True \
#   #               --test_wav $NOISY_WAVNAME --save_clean_path $SAVE_PATH
# done

i=0
for e in ${array[@]}; do
    # echo "${e}"
    python main.py --init_noise_std 0. --save_path segan_v1.1 \
                  --batch_size 100 --g_nl prelu --weights SEGAN-41700 \
                  --preemph 0.99 --bias_deconv True \
                  --bias_downconv True --bias_D_conv True \
                  --test_wav "${e}" --save_clean_path $SAVE_PATH
    let i++
done