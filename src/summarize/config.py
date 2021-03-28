
# 数据集
train_data_path = "./dataset/cnn-dailymail/finished_files/chunked/train_*"
eval_data_path = "./dataset/cnn-dailymail/finished_files/val.bin"
decode_data_path = "./dataset/cnn-dailymail/finished_files/test.bin"
vocab_path = "./dataset/cnn-dailymail/finished_files/vocab"

hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size = 50000
