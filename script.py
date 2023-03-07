import json
import requests
import subprocess
from fastapi import FastAPI, Request, Body
from fastapi.testclient import TestClient
from modules.script_callbacks import on_app_started

client = None

controlnet_models = {
    "canny": "diff_control_sd15_canny_fp16 [ea6e3b9c]",
    "depth": "diff_control_sd15_depth_fp16 [978ef0a1]",
    "hed": "diff_control_sd15_hed_fp16 [86db5d7c]",
    "mlsd": "diff_control_sd15_mlsd_fp16 [14f0845a]",
    "normal": "diff_control_sd15_normal_fp16 [00173cc1]",
    "openpose": "diff_control_sd15_openpose_fp16 [1723948e]",
    "scribble": "diff_control_sd15_scribble_fp16 [1f29174d]",
    "seg": "diff_control_sd15_seg_fp16 [a1e85e27]"
}

def healthcheck():
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True
    return {"state": "healthy", "gpu": gpu}

async def inference(request: Request):
    global client
    body = await request.body()
    model_input = json.loads(body)

    params = None
    mode = 'default'

    if 'endpoint' in model_input:
        endpoint = model_input['endpoint']
        if 'params' in model_input:
            params = model_input['params']
    else:
        mode = 'banana_compat'
        endpoint = 'txt2img'
        params = model_input

    if endpoint == 'controlnet':
        params['controlnet_units'] = [{
            "model": controlnet_models[params['controlnet_model']],
            "input_image": params['controlnet_image'],
            "module": params['controlnet_model'],
        }]
        params_keys = list(params.keys())
        for key in params_keys:
            if key.startswith('controlnet_') and key != 'controlnet_units':
                if (key != 'controlnet_model' and key != 'controlnet_image'):
                    params['controlnet_units'][0][key[11:]] = params[key]
                del params[key]

    if endpoint == 'txt2img' or endpoint == 'controlnet':
        if 'num_inference_steps' in params:
            params['steps'] = params['num_inference_steps']
            del params['num_inference_steps']
        if 'guidance_scale' in params:
            params['cfg_scale'] = params['guidance_scale']
            del params['guidance_scale']

    url = '/sdapi/v1/' + endpoint

    if endpoint == 'controlnet':
        url = '/controlnet/txt2img'

    if params is not None:
        response = client.post(url, json = params)
    else:
        response = client.get(url)

    output = response.json()

    if mode == 'banana_compat' and 'images' in output:
        output = {
            "base64_output": output["images"][0]
        }

    return output

def register_endpoints(block, app):
    global client
    app.add_api_route('/healthcheck', healthcheck, methods=['GET'])
    app.add_api_route('/', inference, methods=['POST'])
    client = TestClient(app)

on_app_started(register_endpoints)
