from dotenv import load_dotenv

load_dotenv()

import os
import uuid
from flask import Flask, send_file, request
from werkzeug.utils import secure_filename

import boto3
from inference.genefacepp_infer import GeneFace2Infer

s3 = boto3.client(
    "s3",
)

S3_BUCKET = os.environ["S3_BUCKET"]


app = Flask(__name__)


@app.route("/ok")
def OK():
    return "OK"


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
    GeneFace2Infer.example_run(params)

    s3.upload_file(output_name, S3_BUCKET, upload_loc)
    return send_file(output_name, download_name="generated.mp4")
