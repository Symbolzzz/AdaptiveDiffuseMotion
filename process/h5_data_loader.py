import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import pdb
import os
import librosa


speaker_id_dict = {
    2: 0,
    4: 1,
    6: 2,
    8: 3,
}


class BEATSpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, style_dim, sequence_length=30*5, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        # TODO：增加 面部特征维度 expression_dim = 51

        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))
        
        expression_mean = np.load(os.path.join(npy_root, "expression_" + dataset + "_mean_" + self.version + ".npy"))
        expression_std = np.load(os.path.join(npy_root, "expression_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [speaker_id_dict[int(self.h5[str(i)]["speaker_id"][:][0])] for i in range(len(self.h5.keys()))]
        self.wav = [self.h5[str(i)]["wav"][:] for i in range(len(self.h5.keys()))]
        self.textraw = [self.h5[str(i)]["textraw"][:] for i in range(len(self.h5.keys()))]
        self.emo = [self.h5[str(i)]["emotion"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.facial = [(self.h5[str(i)]["expression"][:] - expression_mean) / expression_std for i in range(len(self.h5.keys()))]
        
        self.h5.close()
        self.sequence_length = sequence_length
        print("Total clips:", len(self.gesture))
        self.segment_length = sequence_length 

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.gesture[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        
        wav_start = int(start_frame / 30 * 16000)
        wav_end = wav_start + int(self.segment_length / 30 * 16000)
        wav = self.wav[idx][wav_start:wav_end]
        aud = librosa.resample(wav, orig_sr=16000, target_sr=18000)
        mel = librosa.feature.melspectrogram(y=aud, sr=18000, hop_length=1200, n_mels=128)
        mel = mel[..., :total_frame_len]
        mel = np.swapaxes(mel, -1, -2)
        mel = torch.FloatTensor(mel)
        wav = torch.FloatTensor(wav)
        
        textraw = self.textraw[idx][start_frame:end_frame]
        textraw = torch.FloatTensor(textraw)
        
        gesture = self.gesture[idx][start_frame:end_frame]
        # 转化成 tensor
        gesture = torch.FloatTensor(gesture)
        facial = self.facial[idx][start_frame:end_frame]
        facial = torch.FloatTensor(facial)
        
        speaker = np.zeros([self.style_dim])
        # speaker[0] = 1      # dummy speaker
        speaker[self.id[idx]] = 1
        speaker = torch.FloatTensor(speaker)
        
        emo = self.emo[idx][start_frame:end_frame]
        emo = torch.FloatTensor(emo)
        # TODO：增加 面部表情的返回值
        return wav, textraw, gesture, facial, speaker, emo ,mel# [150, dims]
    
    
class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, facial_dim, style_dim, sequence_length=30*5, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        # TODO：增加 面部特征维度 expression_dim = 51
        self.facial_dim = facial_dim
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))
        
        # TODO：增加读取面部特征 均值 方差
        expression_mean = np.load(os.path.join(npy_root, "expression_" + dataset + "_mean_" + self.version + ".npy"))
        expression_std = np.load(os.path.join(npy_root, "expression_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [speaker_id_dict[int(self.h5[str(i)]["speaker_id"][:][0])] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        # TODO：增加读取面部表情 
        self.expression = [(self.h5[str(i)]["expression"][:] - expression_mean) / expression_std for i in range(len(self.h5.keys()))]
        self.h5.close()
        self.sequence_length = sequence_length
        if "v0" in self.version:
            self.gesture_vel = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture]
            self.gesture_acc = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture_vel]
            # TODO：这里加入计算面部表情的速度差和加速度差
            self.expression_vel = [np.concatenate((np.zeros([1, self.facial_dim]), i[1:] - i[:-1]), axis=0) for i in self.expression]
            self.expression_acc = [np.concatenate((np.zeros([1, self.facial_dim]), i[1:] - i[:-1]), axis=0) for i in self.expression_vel]
        print("Total clips:", len(self.gesture))
        self.segment_length = sequence_length 

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        # # TODO：增加读取 面部表情 exprestart = self.expression[idx][start_frame:end_frame]
        exprestart = self.expression[idx][start_frame:end_frame]
        
        if "v0" in self.version:
            vel = self.gesture_vel[idx][start_frame:end_frame]
            acc = self.gesture_acc[idx][start_frame:end_frame]
            gesture = np.concatenate((posrat, vel, acc), axis=-1)
        else:
            gesture = posrat
        vel = self.gesture_vel[idx][start_frame:end_frame]
        acc = self.gesture_acc[idx][start_frame:end_frame]
        gesture = np.concatenate((posrat, vel, acc), axis=-1)
        # gesture = posrat
        # TODO：增加 concatenate 面部表情 速度 和 加速度
        ex_vel = self.expression_vel[idx][start_frame:end_frame]
        ex_acc = self.expression_acc[idx][start_frame:end_frame]
        expression = np.concatenate((exprestart, ex_vel, ex_acc), axis=-1)
        # expression = exprestart
        # gesture = posrat
        
        # 转化成 tensor
        gesture = torch.FloatTensor(gesture)
        expression = torch.FloatTensor(expression)
        speaker = np.zeros([self.style_dim])
        # speaker[0] = 1      # dummy speaker
        speaker[self.id[idx]] = 1
        speaker = torch.FloatTensor(speaker)
        # TODO：增加 面部表情的返回值
        return textaudio, gesture, expression, speaker # [150, dims]


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))


if __name__ == '__main__':
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset("../../process/BEAT_v0.h5", motion_dim=684, facial_dim=51, style_dim=2)

    train_loader = DataLoader(trn_dataset, num_workers=4,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=128,
                              pin_memory=True,
                              drop_last=False)

    for batch_i, batch in enumerate(train_loader, 0):
        textaudio, gesture, expression, speaker = batch     # (128, 150, 1435), (128, 150, 744), (128, 17)
        print(batch_i)
        pdb.set_trace()
