from dotenv import load_dotenv

load_dotenv()

import multiprocessing
import os
import subprocess
import time
import typing
import uuid

import boto3
from flask import Flask, jsonify, request, send_file
from marshmallow import Schema, ValidationError, fields
from werkzeug.utils import secure_filename

from inference.genefacepp_infer import GeneFace2Infer
from tasks.run import run_task

s3 = boto3.client(
    "s3",
)

S3_BUCKET = os.environ["S3_BUCKET"]


app = Flask(__name__)


@app.route("/status")
def status():
    return "OK"


def hparams_from_name(name: str):
    hparams_base = {
        "accumulate_grad_batches": 1,
        "add_eye_blink_cond": True,
        "ambient_coord_dim": 3,
        "ambient_loss_mode": "mae",
        "amp": True,
        "base_config": ["./lm3d_radnerf.yaml"],
        "binary_data_dir": "data/binary/videos",
        "bound": 1,
        "camera_offset": [0, 0, 0],
        "camera_scale": 4.0,
        "clip_grad_norm": 0.0,
        "clip_grad_value": 0,
        "cond_dropout_rate": 0.0,
        "cond_out_dim": 64,
        "cond_type": "idexp_lm3d_normalized",
        "cond_win_size": 1,
        "cuda_ray": True,
        "debug": False,
        "density_thresh": 10,
        "density_thresh_torso": 0.01,
        "desired_resolution": 2048,
        "dt_gamma": 0.00390625,
        "eval_max_batches": 100,
        "exp_name": f"motion2video_nerf/{name}_head",
        "eye_blink_dim": 8,
        "far": 0.9,
        "finetune_lips": True,
        "finetune_lips_start_iter": 80000,
        "geo_feat_dim": 128,
        "grid_interpolation_type": "linear",
        "grid_size": 128,
        "grid_type": "tiledgrid",
        "gui_fovy": 21.24,
        "gui_h": 512,
        "gui_max_spp": 1,
        "gui_radius": 3.35,
        "gui_w": 512,
        "hidden_dim_ambient": 128,
        "hidden_dim_color": 128,
        "hidden_dim_sigma": 128,
        "individual_embedding_dim": 4,
        "individual_embedding_num": 13000,
        "infer": False,
        "infer_audio_source_name": "",
        "infer_bg_img_fname": "",
        "infer_c2w_name": "",
        "infer_cond_name": "",
        "infer_lm3d_clamp_std": 1.5,
        "infer_lm3d_lle_percent": 0.25,
        "infer_lm3d_smooth_sigma": 0.0,
        "infer_out_video_name": "",
        "infer_scale_factor": 1.0,
        "infer_smo_std": 0.0,
        "infer_smooth_camera_path": True,
        "infer_smooth_camera_path_kernel_size": 7,
        "init_method": "tcp",
        "lambda_ambient": None,
        "lambda_dual_fm": 0.0,
        "lambda_lap_ambient_loss": 0.0,
        "lambda_lpips_loss": 0.001,
        "lambda_weights_entropy": 0.0001,
        "load_ckpt": "",
        "load_imgs_to_memory": False,
        "log2_hashmap_size": 16,
        "lpips_mode": "vgg19_v2",
        "lpips_start_iters": 80000,
        "lr": 0.0005,
        "lr_lambda_ambient": 0.01,
        "max_ray_batch": 4096,
        "max_steps": 16,
        "max_updates": 120_000,
        "min_near": 0.05,
        "n_rays": 65536,
        "near": 0.3,
        "nerf_keypoint_mode": "lm68",
        "not_save_modules": ["criterion_lpips", "dual_disc"],
        "num_ckpt_keep": 1,
        "num_layers_ambient": 3,
        "num_layers_color": 2,
        "num_layers_sigma": 3,
        "num_sanity_val_steps": 2,
        "num_steps": 16,
        "num_valid_plots": 5,
        "optimizer_adam_beta1": 0.9,
        "optimizer_adam_beta2": 0.999,
        "polygon_face_mask": True,
        "print_nan_grads": False,
        "processed_data_dir": "data/processed/videos",
        "raw_data_dir": "data/raw/videos",
        "resume_from_checkpoint": 0,
        "save_best": True,
        "save_codes": ["tasks", "modules", "egs"],
        "save_gt": True,
        "scheduler": "exponential",
        "seed": 9999,
        "smo_win_size": 3,
        "smooth_lips": False,
        "sr_start_iters": 0,
        "start_rank": 0,
        "target_ambient_loss": 1e-08,
        "task_cls": "tasks.radnerfs.radnerf_sr.RADNeRFTask",
        "tb_log_interval": 100,
        "torso_head_aware": False,
        "torso_individual_embedding_dim": 8,
        "torso_shrink": 0.8,
        "update_extra_interval": 16,
        "upsample_steps": 0,
        "use_window_cond": True,
        "val_check_interval": 2000,
        "valid_infer_interval": 10000,
        "valid_monitor_key": "val_loss",
        "valid_monitor_mode": "min",
        "validate": False,
        "video_id": name,
        "warmup_updates": 0,
        "weight_decay": 0,
        "with_att": True,
        "with_sr": True,
        "work_dir": f"checkpoints/motion2video_nerf/{name}_head",
        "world_size": -1,
        "zero_dummy": True,
    }
    return hparams_base


TrainSchema = Schema.from_dict(
    {
        "name": fields.String(required=True),
        "video_url": fields.URL(required=True),
        "use_torso": fields.Boolean(dump_default=False),
    }
)


def run_training(name: str):
    global hparams
    hparams = hparams_from_name(name)
    run_task()


@app.route("/train", methods=["POST"])
def start_training():
    request_data = request.get_json()
    schema = TrainSchema()
    try:
        result = schema.load(request_data)
        name = result["name"]
        video_file_name = f"data/raw/videos/{name}.mp4"
        with open(video_file_name, "wb") as f:
            s3.download_fileobj(S3_BUCKET, result["video_url"], f)
        subprocess.run(
            args=f"cp -r ./egs/datasets/May ./egs/datasets/{name}",
            check=True,
            shell=True,
        )
        subprocess.run(
            args=f'for filename in "egs/datasets/{name}/*"; do sed -i \'\' -e "s/May/{name}/Ig" $filename; done',
            check=True,
            shell=True,
        )
        subprocess.run(
            args=f"bash ./data_gen/runs/nerf/run.sh {name}",
            env={"VIDEO_ID": name},
            check=True,
            shell=True,
        )

        proc = multiprocessing.Process(target=run_task)
        proc.join()
        return jsonify()
    except ValidationError as err:
        # Return a nice message if validation fails
        return jsonify(err.messages), 400
    except Exception as err:
        return jsonify(err), 400


def run_inference(params: typing.Dict, name: str, upload_loc: str, output_name: str):
    global hparams
    hparams = hparams_from_name(name)
    GeneFace2Infer.example_run(params)
    s3.upload_file(output_name, S3_BUCKET, upload_loc)


@app.route("/inference/", methods=["POST"])
def start_inference():
    request_data = request.get_json()
    name = request_data["model_name"]
    audio_loc = request_data["audio_loc"]
    instance_uuid = uuid.uuid4()
    dir_base = f"./data/{instance_uuid}"
    inference_base = f"{dir_base}/inference"
    os.mkdir(dir_base)
    os.mkdir(inference_base)
    output_name = f"{inference_base}/{name}_result.mp4"
    audio_file_name = f"{dir_base}/audio.mp4"
    upload_loc = f"./inference/{instance_uuid}/result.mp4"
    with open(audio_file_name, "wb") as f:
        s3.download_fileobj(S3_BUCKET, audio_loc, f)
    params = {
        "a2m_ckpt": "checkpoints/audio2motion_vae",
        "postnet_ckpt": "",
        "head_ckpt": f"./checkpoints/motion2video_nerf/{name}_head",
        "torso_ckpt": f"./checkpoints/motion2video_nerf/{name}_torso",
        "drv_audio_name": audio_file_name,
        "drv_pose": "nearest",
        "blink_mode": "period",
        "temperature": 0.2,
        "mouth_amp": 0.4,
        "lle_percent": 0.2,
        "debug": False,
        "out_name": output_name,
        "raymarching_end_threshold": 0.01,
        "low_memory_usage": False,
    }
    proc = multiprocessing.Process(
        target=run_inference,
        kwargs={
            "params": params,
            "name": name,
            "upload_loc": upload_loc,
            "output_name": output_name,
        },
    )
    proc.join()
    return jsonify({"url": upload_loc, "completed_at": time.time()})
