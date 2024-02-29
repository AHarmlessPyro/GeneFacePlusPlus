from dotenv import load_dotenv

load_dotenv()

import multiprocessing
import os
import subprocess
import tarfile
import time
import typing
from pathlib import Path
import traceback

import boto3
import runpod
import yaml
from marshmallow import EXCLUDE, Schema, fields
from utils.commons.hparams import set_hparams

s3 = boto3.client(
    "s3",
)

S3_BUCKET = os.environ["S3_BUCKET"]
DEBUG = bool(os.environ.get("RUN_DEBUG", False))

BASE_DIR = os.getcwd()

os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "./")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.makedirs("data/raw/videos", exist_ok=True)


def cp_configs(name: str, head_kwargs: typing.Dict, torso_kwargs: typing.Dict):
    def get_new_path(file_path):
        *base_path, file = file_path.split(os.path.sep)
        new_base_path = os.path.join(*base_path).replace("May", name)
        return new_base_path, os.path.join(new_base_path, file)

    def update_yml(file: str, kwargs: typing.Dict):
        _, new_file_name = get_new_path(file)
        with open(file) as f_read:
            with open(new_file_name, "w+") as f_write:
                text = f_read.read()
                new_text = text.replace("May", name)
                yml_text = yaml.load(new_text, Loader=yaml.Loader)
                yml_text.update(**kwargs)
                stringified = yaml.dump(yml_text, Dumper=yaml.Dumper)
                f_write.write(stringified)

    def copy_yml(file: str):
        new_base_path, new_file_name = get_new_path(file)
        os.makedirs(new_base_path, exist_ok=True)
        with open(file) as f_read:
            with open(new_file_name, "w+") as f_write:
                text = f_read.read()
                f_write.write(text.replace("May", name))

    def copy_head(head_kwargs: typing.Dict):
        copy_yml("./egs/datasets/May/lm3d_radnerf.yaml")
        update_yml("./egs/datasets/May/lm3d_radnerf_sr.yaml", head_kwargs)

    def copy_torso(torso_kwargs: typing.Dict):
        copy_yml("./egs/datasets/May/lm3d_radnerf_torso.yaml")
        update_yml(
            "./egs/datasets/May/lm3d_radnerf_torso_sr.yaml",
            torso_kwargs,
        )

    copy_head(head_kwargs)
    copy_torso(torso_kwargs)


def compile_response(
    input: typing.Dict,
    task_type: typing.Literal["train"] | typing.Literal["infer"],
    artifact_upload_path: str,
    **kwargs,
):
    return {
        "type": task_type,
        "taskId": input["id"],
        "orgId": input["org_id"],
        "completedAt": int(time.time()),
        "artifactUploadPath": artifact_upload_path,
        "data": kwargs,
        "secretValue": input["secretValue"],
    }


TrainSchema = Schema.from_dict(
    {
        "id": fields.UUID(required=True),
        "orgId": fields.UUID(required=True),
        "secretValue": fields.String(required=True),
        "name": fields.String(required=True),
        "videoPath": fields.Str(required=True),
        "useTorso": fields.Boolean(load_default=False),
        "trainParams": fields.Dict(keys=fields.Str(), values=fields.Raw()),
    }
)


def run_training(name: str, train_torso: bool = True, **params):
    from tasks.run import run_task

    raise Exception("frick")

    base_env = os.environ.copy()
    base_env["VIDEO_ID"] = name

    subprocess.run(
        args=f"bash --login ./data_gen/runs/nerf/run.sh {name}",
        env=base_env,
        check=True,
        shell=True,
    )

    set_hparams(
        config=f"./egs/datasets/{name}/lm3d_radnerf_sr.yaml",
        exp_name=f"motion2video_nerf/{name}_head",
    )
    run_task()
    if train_torso:
        set_hparams(
            config=f"./egs/datasets/{name}/lm3d_radnerf_torso_sr.yaml",
            exp_name=f"motion2video_nerf/{name}_torso",
            hparams_str=f"head_model_dir=checkpoints/motion2video_nerf/{name}_head",
        )
        run_task()


def start_training(request_data: typing.Dict):
    schema = TrainSchema(unknown=EXCLUDE)
    result = schema.load(
        request_data,
    )

    name = result["name"]
    instance_uuid = result["id"]
    video_path = result["videoPath"]
    train_params = result["trainParams"]
    train_torso = result["useTorso"]
    video_file_name = f"data/raw/videos/{name}.mp4"

    print("Unpacked request params")

    os.makedirs(f"./checkpoints/motion2video_nerf/{name}_head", exist_ok=True)
    if train_torso:
        os.makedirs(f"./checkpoints/motion2video_nerf/{name}_torso", exist_ok=True)

    cp_configs(name, train_params, train_params)

    print("Copied config")

    with open(video_file_name, "wb") as f:
        s3.download_fileobj(S3_BUCKET, video_path, f)

    print("Downloaded resources")

    proc = multiprocessing.Process(
        target=run_training, args=[name, train_torso], kwargs=train_params
    )
    proc.start()
    proc.join()

    print(proc)
    print("Completed subprocess")

    archive_name = f"{name}.tar"
    base_path = BASE_DIR
    tar_location = Path(os.path.join(base_path, archive_name))

    upload_loc = f"training/{instance_uuid}/data.tar"
    tar_archive = tarfile.open(str(tar_location), "w:")
    tar_archive.add(f"./data/binary/videos/{name}/trainval_dataset.npy")
    tar_archive.add(f"./data/processed/videos/{name}/")
    tar_archive.add(f"./checkpoints/motion2video_nerf/{name}_head")
    if train_torso:
        tar_archive.add(f"./checkpoints/motion2video_nerf/{name}_torso")
    tar_archive.close()
    s3.upload_file(tar_location, S3_BUCKET, upload_loc)

    print("Uploaded tar")
    return (
        {
            "refresh_worker": False,
            "job_results": compile_response(result, "train", upload_loc),
        },
    )


InferSchema = Schema.from_dict(
    {
        "id": fields.UUID(required=True),
        "orgId": fields.UUID(required=True),
        "secretValue": fields.String(required=True),
        "name": fields.String(required=True),
        "audioPath": fields.Str(required=True),
        "dataPath": fields.Str(required=True),
        "useTorso": fields.Boolean(load_default=True),
        "trainParams": fields.Dict(
            keys=fields.Str(), values=fields.Raw(), load_default={}
        ),
    }
)


def run_inference(
    params: typing.Dict, name: str, upload_loc: str, output_name: str, **train_params
):
    from inference.genefacepp_infer import GeneFace2Infer

    GeneFace2Infer.example_run(params)
    s3.upload_file(output_name, S3_BUCKET, upload_loc)


def start_inference(request_data: typing.Dict):
    schema = InferSchema(unknown=EXCLUDE)
    result = schema.load(request_data)

    instance_uuid = result["id"]
    name = result["name"]
    audio_loc = result["audioPath"]
    data_loc = result["dataPath"]
    use_torso = result["useTorso"]
    train_params: typing.Dict = result["trainParams"]

    dir_base = f"./data/{instance_uuid}"
    inference_base = f"{dir_base}/inference"
    output_name = f"{inference_base}/{name}_result.mp4"
    audio_file_name = f"{dir_base}/audio.mp4"
    upload_loc = f"inference/{instance_uuid}/result.mp4"

    cp_configs(name, train_params, train_params)

    os.makedirs(f"./checkpoints/motion2video_nerf/{name}_head", exist_ok=True)
    os.makedirs(f"./checkpoints/motion2video_nerf/{name}_torso", exist_ok=True)

    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(inference_base, exist_ok=True)

    with open(audio_file_name, "wb") as f:
        s3.download_fileobj(S3_BUCKET, audio_loc, f)

    tarfile_loc = "data.tar"

    with open(tarfile_loc, "wb") as f:
        s3.download_fileobj(S3_BUCKET, data_loc, f)

    file = tarfile.open(tarfile_loc)
    file.extractall(BASE_DIR)

    params = {
        "a2m_ckpt": train_params.get("a2m_ckpt", "checkpoints/audio2motion_vae"),
        "postnet_ckpt": train_params.get("postnet_ckpt", ""),
        "head_ckpt": train_params.get(
            "head_ckpt", f"./checkpoints/motion2video_nerf/{name}_head"
        ),
        "torso_ckpt": train_params.get(
            "torso_ckpt",
            f"./checkpoints/motion2video_nerf/{name}_torso" if use_torso else "",
        ),
        "drv_audio_name": train_params.get("drv_audio_name", audio_file_name),
        "drv_pose": train_params.get("drv_pose", "nearest"),
        "blink_mode": train_params.get("blink_mode", "period"),
        "temperature": train_params.get("temperature", 0.2),
        "mouth_amp": train_params.get("mouth_amp", 0.4),
        "lle_percent": train_params.get("lle_percent", 0.2),
        "debug": train_params.get("debug", False),
        "out_name": train_params.get("out_name", output_name),
        "raymarching_end_threshold": train_params.get(
            "raymarching_end_threshold", 0.01
        ),
        "low_memory_usage": train_params.get("low_memory_usage", False),
    }

    proc = multiprocessing.Process(
        target=run_inference,
        kwargs={
            "params": params,
            "name": name,
            "upload_loc": upload_loc,
            "output_name": output_name,
            "train_params": train_params,
        },
    )
    proc.start()
    proc.join()

    return {
        "refresh_worker": True,
        "job_results": compile_response(result, "infer", upload_loc),
    }


def process(job):
    try:
        print("Got incoming request in handler")
        input = job.get("input", {})
        task_type = input.get("task_type", None)
        if task_type is None:
            raise Exception("No task type found")
        elif task_type == "train":
            print("Got request of type training")
            return start_training(input)
        elif task_type == "infer":
            print("Got request of type inference")
            return start_inference(input)
        else:
            Exception(f"No task of type {task_type} found")
    except:
        traceback.print_exc()
        raise 

if __name__ == "__main__":
    runpod.serverless.start({"handler": process})
