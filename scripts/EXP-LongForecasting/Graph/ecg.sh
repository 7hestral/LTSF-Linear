
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1000
model_name=GraphDLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ecg \
  --model_id ecg_$seq_len'_'1000 \
  --model $model_name \
  --freq ms \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 1000 \
  --enc_in 5 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'ecg_$seq_len'_'1000.log