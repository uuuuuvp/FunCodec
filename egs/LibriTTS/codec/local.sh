#!/usr/bin/env bash

. ./path.sh || exit 1;

export PYTHONPATH=/mnt/e/fun/FunCodec/funcodec:$PYTHONPATH

# machines configuration
gpu_devices="0"
gpu_num=1
count=1

# general configuration
feats_dir="."
exp_dir="."
dumpdir=dump/LibriTTS
stage=1
stop_stage=5
corpus_dir=corpus/LibriTTS

# training related
tag=""
train_set=train
valid_set=dev
train_config=conf/freqcodec_mag_angle_16k_n32_600k_step.yaml
init_param=
state_dir=LibriTTS_states

# inference related
inference_model=30epoch.pth
inference_tag="inference"
batch_size=1
test_sets="test-clean"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
need_indices=false
need_sub_quants=false
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
infer_cmd=utils/run.pl
sample_frequency=16000
file_sampling_rate=16000
bit_width=4000
use_scale=false
use_ppg=false
model_dir=

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ -z "${model_dir}" ]; then
  model_dir="$(basename "${train_config}" .yaml)${tag}"
fi

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi


# Data collecting
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: collecting data sets."
  mkdir -p ${dumpdir}/train_24k ${dumpdir}/dev_24k

  for name in train-clean-100 train-clean-360; do
    echo "collecting ${name} in to ${dumpdir}/train_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/train_24k/wav.scp
  done

  for name in dev-clean; do
    echo "collecting ${name} in to ${dumpdir}/dev_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort >> ${dumpdir}/dev_24k/wav.scp
  done

  for name in test-clean; do
    mkdir -p ${dumpdir}/${name}_24k
    echo "collecting ${name} in to ${dumpdir}/${name}_24k/wav.scp"
    find ${corpus_dir}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort > ${dumpdir}/${name}_24k/wav.scp
  done
fi

# Dump data to ark and convert it to the sampling rate of 16000
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Dump data to ark."
  for name in train dev; do
    torchrun --nproc_per_node=1 --master_port=1234 scripts/dump_to_wav_ark.py \
      --wav_scp ${dumpdir}/${name}_24k/wav.scp \
      --out_dir ${dumpdir}/${name}/arks \
      --sample_rate 16000

    mkdir -p ${dumpdir}/${name} exp/${state_dir}/${name}
    cat ${dumpdir}/${name}/arks/wav.*.scp | sort > ${dumpdir}/${name}/wav.scp
    cat ${dumpdir}/${name}/arks/length.*.txt | shuf > exp/${state_dir}/${name}/speech_shape
  done
fi

# Training Stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training"
    mkdir -p ${exp_dir}/exp/${model_dir}/log
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funcodec.bin.codec_train \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/wav.scp,speech,kaldi_ark \
                --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
                --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/wav.scp,speech,kaldi_ark \
                --valid_shape_file ${feats_dir}/exp/${state_dir}/${valid_set}/speech_shape \
                --output_dir ${exp_dir}/exp/${model_dir} \
                --config $train_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --multiprocessing_distributed true \
                --dist_world_size $gpu_num \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Testing Stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Inference @ ${bit_width}"
    for dset in ${test_sets}; do
        echo "Processing for $dset @ ${bit_width}"
        asr_exp=${exp_dir}/exp/${model_dir}
        _dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
        mkdir -p "${_dir}/logdir"
        _data="${feats_dir}/${dumpdir}/${dset}"
    done
fi
