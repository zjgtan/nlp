import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size,
                            hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, enc_input, enc_len):
        embedded = self.embedding(enc_input)
        packed = pack_padded_sequence(embedded, enc_len, batch_first=True)
        output, (hT, cT) = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        return encoder_outputs

class AttDecoder(nn.Module):
    def __init__(self, vocab_size, dec_embedding_size, dec_hidden_size):
        super(AttDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dec_embedding_size)
        self.lstm = nn.LSTM(dec_embedding_size, dec_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(dec_hidden_size, dec_hidden_size)
        self.fc2 = nn.Linear(dec_hidden_size, vocab_size)

    def forward(self, y_in, h_t, c_t):
        y_in_embedding = self.embedding(y_in)
        output, (h_t, c_t) = self.lstm(y_in_embedding, (h_t, c_t))

        fc = F.relu(self.fc1(output))
        fc = F.softmax(self.fc2(fc))

        return fc, h_t, c_t

class PointerNetWork(nn.Module):
    def __init__(self, enc_vocab_size, enc_embedding_size, enc_hidden_size,
                 dec_vocab_size, dec_embedding_size, dec_hidden_size):
        super(PointerNetWork, self).__init__()
        self.encoder = Encoder(enc_vocab_size, enc_embedding_size, enc_hidden_size)
        self.decoder = AttDecoder(dec_vocab_size, dec_embedding_size, dec_hidden_size)

    def forward(self, enc_input,
                enc_len,
                dec_input=None,
                dec_len=None,
                dec_output=None,
                is_train=False):
        """
        :param enc_input: 经过padding的编码器输入文本序列
        :param dec_input: 经过padding的解码器输入文本序列
        :param dec_output: 经过padding的解码器目标序列
        :param is_train: 训练还是解码
        :return:
        """
        enc_output, enc_hT, enc_cT = self.encoder(enc_input, enc_len)

        if is_train:
            h_t, c_t = enc_hT, enc_cT
            for step in range(max(dec_len)):
                y_in = dec_input[:, step]
                dec_preds, h_t, c_t = self.decoder(y_in, h_t, c_t)


        return enc_output

if __name__ == '__main__':
    net = PointerNetWork(enc_vocab_size=10, enc_embedding_size=2, enc_hidden_size=2)
    enc_input = torch.LongTensor([[1, 2, 3]])
    enc_len = [3]
    print(net(enc_input, enc_len))



