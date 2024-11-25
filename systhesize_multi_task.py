from model.ADM import AdaptiveDiffuseMotion
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
import os
import copy
import numpy as np
import yaml
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process.process_BEAT_bvh import wavlm_init, pose2bvh, pose2bvh_bugfix, expre2json
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


speaker_id_dict = {
    2: 0,
    4: 1,
    6: 2,
    8: 3,
}

id_speaker_dict = {
    0: 2,
    1: 4,
    2: 6,
    3: 8,
}


def inference(
    args,
    save_dir,
    prefix,
    textaudio,
    sample_fn,
    model,
    n_frames=0,
    smoothing=False,
    skip_timesteps=0,
    style=None,
    seed=123456,
    dataset="BEAT",
):
    torch.manual_seed(seed)
    if dataset == "BEAT":
        # speaker_id = speaker_id_dict[int(prefix.split('_')[0])]
        speaker_id = 0
        speaker = id_speaker_dict[speaker_id]
        style = np.zeros([args.style_dim])
        style[speaker_id] = 1
    elif dataset == "TWH":
        speaker = np.where(style == np.max(style))[0][0]
    if n_frames == 0:
        n_frames = textaudio.shape[0]
    else:
        textaudio = textaudio[:n_frames]
    real_n_frames = copy.deepcopy(n_frames)  # 1830
    stride_poses = args.n_poses - args.n_seed
    if n_frames < stride_poses:
        num_subdivision = 1
        n_frames = stride_poses
    else:
        num_subdivision = math.ceil(n_frames / stride_poses)
        n_frames = num_subdivision * stride_poses
        print(
            "real_n_frames: {}, num_subdivision: {}, stride_poses: {}, n_frames: {}, speaker_id: {}".format(
                real_n_frames,
                num_subdivision,
                stride_poses,
                n_frames,
                np.where(style == np.max(style))[0][0],
            )
        )

    model_kwargs_ = {"y": {}}
    model_kwargs_["y"]["mask"] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
    model_kwargs_["y"]["style"] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_["y"]["mask_local"] = torch.ones(1, args.n_poses).bool().to(mydevice)

    textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim]).to(
        mydevice
    )
    textaudio = torch.cat((textaudio, textaudio_pad), 0)
    audio_reshape = textaudio.reshape(
        num_subdivision, stride_poses, args.audio_feature_dim
    ).transpose(0, 1)

    if dataset == "BEAT":
        data_mean_ = np.load("./process/gesture_BEAT_mean_" + args.version + ".npy")
        data_std_ = np.load("./process/gesture_BEAT_std_" + args.version + ".npy")
        # NOTE：在此处加入读取面部表情 均值 和 方差 的代码
        data_mean_ex = np.load(
            "./process/expression_BEAT_mean_" + args.version + ".npy"
        )
        data_std_ex = np.load("./process/expression_BEAT_std_" + args.version + ".npy")

    data_mean = np.array(data_mean_)
    data_std = np.array(data_std_)

    data_mean_ex = np.array(data_mean_ex)
    data_std_ex = np.array(data_std_ex)

    shape_ = (1, model.njoints + model.nexpressions, model.nfeats, args.n_poses)
    out_list = []
    for i in range(0, num_subdivision):
        print(i, num_subdivision)
        model_kwargs_["y"]["audio"] = audio_reshape[:, i : i + 1]
        if i == 0:
            model_kwargs_["y"]["audio"] = model_kwargs_["y"]["audio"].transpose(0, 1)
            # NOTE：增加面部表情部分
            if speaker == 2:
                seed_gesture = np.load("./seeds/gesture" + "/2_scott_0_9_9.npy")[
                    : args.n_seed + 2
                ]  # any speaker, here we only use seed pose of 2_scott_0_1_1.npy
                seed_expression = np.load("./seeds/expression" + "/2_scott_0_9_9.npy")[
                    : args.n_seed + 2
                ]
            elif speaker == 4:
                seed_gesture = np.load("./seeds/gesture" + "/4_lawrence_0_9_9.npy")[
                    : args.n_seed + 2
                ]
                seed_expression = np.load(
                    "./seeds/expression" + "/4_lawrence_0_9_9.npy"
                )[: args.n_seed + 2]
            elif speaker == 6:
                seed_gesture = np.load("./seeds/gesture" + "/6_carla_0_9_9.npy")[
                    : args.n_seed + 2
                ]
                seed_expression = np.load("./seeds/expression" + "/6_carla_0_9_9.npy")[
                    : args.n_seed + 2
                ]
            elif speaker == 8:
                seed_gesture = np.load("./seeds/gesture" + "/8_catherine_0_9_9.npy")[
                    : args.n_seed + 2
                ]
                seed_expression = np.load(
                    "./seeds/expression" + "/8_catherine_0_9_9.npy"
                )[: args.n_seed + 2]
            else:
                raise NotImplementedError

            seed_gesture = (seed_gesture - data_mean) / data_std
            seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
            seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
            seed_gesture_ = np.concatenate(
                (seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1
            )  # (args.n_seed, args.njoints)
            seed_gesture_ = (
                torch.from_numpy(seed_gesture_)
                .float()
                .transpose(0, 1)
                .unsqueeze(0)
                .to(mydevice)
            )
            # seed_gesture_ = seed_gesture_.permute(0, 2, 1)
            print(seed_gesture_.shape)
            # NOTE：需要在这里做表情手势特征融合，读取训练的融合模型
            seed_expression = (seed_expression - data_mean_ex) / data_std_ex
            seed_expression_vel = seed_expression[1:] - seed_expression[:-1]
            seed_expression_acc = seed_expression_vel[1:] - seed_expression_vel[:-1]
            seed_expression_ = np.concatenate(
                (seed_expression[2:], seed_expression_vel[1:], seed_expression_acc),
                axis=1,
            )
            seed_expression_ = (
                torch.from_numpy(seed_expression_)
                .float()
                .transpose(0, 1)
                .unsqueeze(0)
                .to(mydevice)
            )
            # seed_expression_ = seed_expression_.permute(0, 2, 1)

            model_kwargs_["y"]["seed gesture"] = seed_gesture_.unsqueeze(2)
            model_kwargs_["y"]["seed expression"] = seed_expression_.unsqueeze(2)

        else:
            model_kwargs_["y"]["audio"] = model_kwargs_["y"]["audio"].transpose(0, 1)

            model_kwargs_["y"]["seed gesture"] = out_list[-1][
                :, :2052, :, -args.n_seed :
            ].to(mydevice)
            model_kwargs_["y"]["seed expression"] = out_list[-1][
                :, 2052:, :, -args.n_seed :
            ].to(mydevice)

        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )
        # smoothing motion transition
        if len(out_list) > 0 and args.n_seed != 0:
            last_poses = out_list[-1][
                ..., -args.n_seed :
            ]  # # (1, model.njoints, 1, args.n_seed)
            out_list[-1] = out_list[-1][..., : -args.n_seed]  # delete last 4 frames
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                next = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
        out_list.append(sample)  # [1, 2205, 1, 150]

    if "v0" in args.version:
        motion_feature_division = 3
    elif "v2" in args.version:
        motion_feature_division = 1
    else:
        raise ValueError("wrong version name")

    # NOTE：这后面要加上面部表情处理的代码，就是从 out_list 中解码出最终的 JSON 格式文件
    out_pose_list = []
    out_expression_list = []

    for i in out_list:
        # Step 3: 分割成两个部分
        part_gesture = i[:, :2052, :, :]  # [1, 2052, 1, 120]
        part_expression = i[:, 2052:, :, :]  # [1, 153, 1, 120]

        out_pose_list.append(part_gesture)
        out_expression_list.append(part_expression)

    out_pose_list = [
        i.detach().data.cpu().numpy()[:, : args.njoints // motion_feature_division]
        for i in out_pose_list
    ]  # [1, 2052, 1, 120]
    # 处理之后 (1, 684, 1, 120)

    if len(out_pose_list) > 1:
        out_dir_vec_1 = np.vstack(out_pose_list[:-1])  # (16, 684, 1, 120)
        sampled_seq_1 = (
            out_dir_vec_1.squeeze(2)
            .transpose(0, 2, 1)
            .reshape(batch_size, -1, model.njoints // motion_feature_division)
        )  # (1, 1920, 684)
        out_dir_vec_2 = (
            np.array(out_pose_list[-1]).squeeze(2).transpose(0, 2, 1)
        )  # (1, 150, 684)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_pose_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = sampled_seq[:, args.n_seed :]

    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    print(out_poses.shape, real_n_frames)
    out_poses = out_poses[:real_n_frames]
    pose2bvh_bugfix(
        save_dir,
        prefix,
        out_poses,
        pipeline="/mnt/disk1/xey/sub_adaptive_diff/process/resource/data_pipe_30fps"
        + "_speaker"
        + str(speaker)
        + ".sav",
    )

    # NOTE：在这里合成 json 文件
    out_expression_list = [
        i.detach().data.cpu().numpy()[:, : args.nexpressions // motion_feature_division]
        for i in out_expression_list
    ]  # [1, 2052, 1, 120]
    if len(out_expression_list) > 1:
        out_dir_vec_1 = np.vstack(out_expression_list[:-1])  # (16, 684, 1, 120)
        sampled_seq_1 = (
            out_dir_vec_1.squeeze(2)
            .transpose(0, 2, 1)
            .reshape(batch_size, -1, model.nexpressions // motion_feature_division)
        )  # (1, 1920, 684)
        out_dir_vec_2 = (
            np.array(out_expression_list[-1]).squeeze(2).transpose(0, 2, 1)
        )  # (1, 150, 684)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_expression_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = sampled_seq[:, args.n_seed :]

    out_expressions = np.multiply(sampled_seq[0], data_std_ex) + data_mean_ex
    print(out_expressions.shape, real_n_frames)  # (2040, 51) 1922
    out_expressions = out_expressions[:real_n_frames]
    expre2json(save_dir, prefix, out_expressions)


def main(
    args,
    save_dir,
    model_path,
    tst_path=None,
    max_len=0,
    skip_timesteps=0,
    tst_prefix=None,
    dataset="BEAT",
    wav_path=None,
    txt_path=None,
    wavlm_path=None,
    word2vector_path=None,
    fusion_model_path=None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model = AdaptiveDiffuseMotion(
        modeltype="",
        njoints=args.njoints,
        nfeats=1,
        nexpressions=args.nexpressions,
        arch="trans_enc",
        latent_dim=args.latent_dim,
        n_seed=args.n_seed,
        cond_mask_prob=args.cond_mask_prob,
        device=device_name,
        style_dim=args.style_dim,
        source_audio_dim=args.audio_feature_dim,
        audio_feat_dim_latent=args.audio_feat_dim_latent,
    )

    diffusion = create_gaussian_diffusion()

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()

    sample_fn = diffusion.p_sample_loop  # predict x_start

    if tst_path is not None:
        tst_audio_dir = os.path.join(tst_path, "audio_" + dataset)
        tst_text_dir = os.path.join(tst_path, "text_" + dataset)

        for i, filename in enumerate(tst_prefix):
            print(f"Processing: {filename}")

            speaker_id = speaker_id_dict[int(filename.split("_")[0])]
            speaker = np.zeros([args.style_dim])
            speaker[speaker_id] = 1

            audio_path = os.path.join(tst_audio_dir, filename + ".npy")
            audio = np.load(audio_path)
            text_path = os.path.join(tst_text_dir, filename + ".npy")
            text = np.load(text_path)
            textaudio = np.concatenate((audio, text), axis=-1)
            textaudio = torch.FloatTensor(textaudio)
            textaudio = textaudio.to(mydevice)

            inference(
                args,
                save_dir,
                filename,
                textaudio,
                sample_fn,
                model,
                n_frames=max_len,
                smoothing=True,
                skip_timesteps=skip_timesteps,
                style=speaker,
                seed=123456,
                dataset=dataset,
            )
    else:
        from process.process_BEAT_bvh import load_wordvectors, load_audio, load_tsv

        wavlm_model, cfg = wavlm_init(wavlm_path, mydevice)
        word2vector = load_wordvectors(fname=word2vector_path)

        wav = load_audio(wav_path, wavlm_model, cfg)
        clip_len = wav.shape[0]
        tsv = load_tsv(txt_path, word2vector, clip_len)
        textaudio = np.concatenate((wav, tsv), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)
        speaker = np.zeros([17])
        speaker[0] = 1  # random choice will be great
        filename = wav_path.split("/")[-1][:-4]
        inference(
            args,
            save_dir,
            filename,
            textaudio,
            sample_fn,
            model,
            n_frames=max_len,
            smoothing=True,
            skip_timesteps=skip_timesteps,
            style=speaker,
            seed=123456,
            dataset=dataset,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaptiveDiffuseMotion")
    parser.add_argument("--config", default="./configs/AdaptiveDiffuseMotion.yml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--tst_prefix", nargs="+")
    parser.add_argument("--no_cuda", type=list, default=["0"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tst_path", type=str, default=None)
    parser.add_argument("--wav_path", type=str, default=None)
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="sample_dir")
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--skip_timesteps", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="BEAT")
    parser.add_argument("--wavlm_path", type=str, default="./WavLM/WavLM-Large.pt")
    parser.add_argument("--word2vector_path", type=str, default="./crawl-300d-2M.vec")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    config = EasyDict(config)

    device_name = "cuda:" + args.gpu
    mydevice = torch.device("cuda:" + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    batch_size = 1

    model_root = config.model_path.split("/")[1]
    model_spicific = config.model_path.split("/")[-1].split(".")[0]
    if config.tst_prefix is not None:
        config.tst_path = "../../" + config.dataset + "_dataset/processed/"

    print(
        "model_root",
        model_root,
        "tst_path",
        config.tst_path,
        "save_dir",
        config.save_dir,
    )

    main(
        config,
        config.save_dir,
        config.model_path,
        tst_path=config.tst_path,
        max_len=config.max_len,
        skip_timesteps=config.skip_timesteps,
        tst_prefix=config.tst_prefix,
        dataset=config.dataset,
        wav_path=config.wav_path,
        txt_path=config.txt_path,
        wavlm_path=config.wavlm_path,
        word2vector_path=config.word2vector_path,
    )
