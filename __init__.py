import uuid

import torch

import boto3

import os
import sys
import json
import hashlib
import traceback
import math
import time

import urllib
import io
import base64

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management

import comfy
from urllib import request
from PIL import Image, ImageOps

import folder_paths
from comfy_extras.chainner_models import model_loading
from custom_nodes.DTGlobalVariables import variables

try:
    from torchvision.transforms import ToPILImage
except ImportError:
    def ToPILImage():
        pass

from custom_nodes.DTAIComfyImageSubmit import config


class RemoteLoader:
    def __init__(self, path, uri):
        self.data = {}
        self.uri = uri
        self.path = path

        self.load()

    def load(self):
        try:
            with urllib.request.urlopen(self.uri) as f:
                self.data = json.loads(f.read().decode('utf-8'))
            print(f"Loaded {self.path} from {self.uri}")
        except Exception as e:
            print(f"Failed to load {self.uri} for {self.path}: {e}")

    def list(self):
        try:
            return list(self.data.keys())
        except AttributeError:
            return []

    def filename(self, key):
        if key not in self.data:
            raise KeyError(f"Key {key} not found in {self.uri}")

        key = self.data[key]

        filename = ""
        # if key is a url with a filename with any extension at the end use it
        # verify this by checking if the key is a url and the last segment includes a .
        if key.startswith("http") and key.split("/")[-1].find(".") > 0:
            filename = key.split("/")[-1]
            # strip off any query params from filename
            filename = filename.split("?")[0]
        else:
            # Use hashlib to create the md5 hash of the key
            hash_object = hashlib.md5(key.encode())
            md5_hash = hash_object.hexdigest()
            filename = md5_hash

        # if download_path is set, use it joined with self.path
        if '' != config.download_path:
            folder = os.path.join(config.download_path, self.path)
        else:
            # otherwise, use the default path
            folder = folder_paths.get_folder_paths(self.path)[0]

        full_path = os.path.join(folder, filename)

        # Combine self.path with md5_hash to get the filepath
        return full_path

    def download(self, key):
        if key not in self.data:
            raise KeyError(f"Key {key} not found in {self.uri}")

        filename = self.filename(key)

        # If the file doesn't exist at that path, download and save it
        if not os.path.exists(filename):
            urllib.request.urlretrieve(self.data[key], filename)

        return filename

upscalers = RemoteLoader("checkpoints", "https://api.aiart.doubtech.com/comfyui/upscalers")
checkpoints = RemoteLoader("checkpoints", "https://api.aiart.doubtech.com/comfyui/checkpoints")
vae = RemoteLoader("vae", "https://api.aiart.doubtech.com/comfyui/vae")
lora = RemoteLoader("loras", "https://api.aiart.doubtech.com/comfyui/lora")
clip = RemoteLoader("clip", "https://api.aiart.doubtech.com/comfyui/clip")
controlNet = RemoteLoader("controlnet", "https://api.aiart.doubtech.com/comfyui/controlnet")
controlNetDiff = RemoteLoader("controlnet", "https://api.aiart.doubtech.com/comfyui/controlnetdiff")
style = RemoteLoader("style_models", "https://api.aiart.doubtech.com/comfyui/style")
clipVision = RemoteLoader("clip_vision", "https://api.aiart.doubtech.com/comfyui/clipvision")
unclipCheckpoint = RemoteLoader("checkpoints", "https://api.aiart.doubtech.com/comfyui/unclip")
gligen = RemoteLoader("gligen", "https://api.aiart.doubtech.com/comfyui/gligen")
hypernetwork = RemoteLoader("hypernetwork", "https://api.aiart.doubtech.com/comfyui/hypernetwork")
configs = RemoteLoader("configs", "https://api.aiart.doubtech.com/comfyui/configs")


class SubmitImage:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "prompt_text": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "tags": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "title": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "alt": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "caption": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "set": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "private": ("INT", {
                    "default": False
                })
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "upload"

    OUTPUT_NODE = True

    CATEGORY = "DoubTech/Image"

    def upload_image_to_s3(self, image, pnginfo=None):
        # Create an S3 client
        s3_client = boto3.client('s3', aws_access_key_id=config.s3_key, aws_secret_access_key=config.s3_secret, region_name=config.s3_region)

        # Convert the image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG', pnginfo=pnginfo)
        img_bytes.seek(0)

        # Generate a UUID-based name for the S3 key
        s3_key = f"images/{uuid.uuid4()}.png"

        # Upload the image to the S3 bucket
        try:
            s3_client.upload_fileobj(img_bytes, config.s3_bucket, s3_key)
            print(f"Image uploaded to S3 bucket: {config.s3_bucket}, key: {s3_key}")

            # Generate the URL of the uploaded image
            s3_url = f"https://{config.s3_bucket}.s3.amazonaws.com/{s3_key}"
            return s3_url
        except Exception as e:
            print("Error uploading image to S3:", e)
            return None


    def upload(self, images, prompt_text, tags, title, alt, caption, set, private, prompt=None, extra_pnginfo=None):
        print("uploading image...")

        if variables.generated_prompt is not None:
            prompt_text = variables.generated_prompt

        # uriencode the parameters
        if "tags" in variables.state:
            current_tags = variables.state["tags"].split(",")
            current_tags = [t for t in current_tags if t != ""]
            tags = tags.split(",") or []
            tags = [t for t in tags if t != ""]
            # Merge the two arrays into one and remove duplicates
            tags = current_tags + [x for x in tags if x not in current_tags]
            tags = ",".join(tags)

        tags = urllib.parse.quote(variables.apply(tags, "tags"))
        title = urllib.parse.quote(variables.apply(title, "title"))
        alt = urllib.parse.quote(variables.apply(alt, "alt"))
        setName = urllib.parse.quote(variables.apply(set, "set"))
        prompt_text = urllib.parse.quote(variables.apply(prompt_text))
        caption = urllib.parse.quote(variables.apply(caption))

        # Create a post request to submit the image as the post body to the backend
        uri = "https://api.aiart.doubtech.com/comfyui/submit?key={}&tags={}&title={}&alt={}&set={}&prompt={}&caption={}&private={}".format(
            config.apikey,
            tags,
            title,
            alt,
            setName,
            prompt_text,
            caption,
            private)

        print(f"Submitting {prompt_text} with data:\n{prompt}")

        #iterate over the images
        for image in images:
            # Convert the image to a png
            i = 255. * image.cpu().numpy()
            png = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            width = png.width
            height = png.height
            uri += "&width={}&height={}".format(width, height)

            if config.use_s3:
                # Upload the image to S3
                s3_url = self.upload_image_to_s3(png, pnginfo=metadata)
                if s3_url is None:
                    return ()
                uri += "&image_url=" + urllib.parse.quote(s3_url)

                print("Submission uri: " + uri)
                print("  (S3 URL: " + s3_url + ")")
                with request.urlopen(request.Request(uri, method='POST'),
                                     timeout=600) as f:
                    print(f.read().decode('utf-8'))
            else:
                img_bytes = io.BytesIO()
                png.save(img_bytes, format='PNG', pnginfo=metadata)
                img_bytes.seek(0)

                # Encode the image bytes to base64
                encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                print("Submission uri: " + uri)
                # Submit the image using a POST request
                with request.urlopen(request.Request(uri, data=encoded.encode('utf-8'), method='POST'), timeout=600) as f:
                    print(f.read().decode('utf-8'))

            return ()


class DTNodeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (checkpoints.list(), ), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = checkpoints.download(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out


class DTVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (vae.list(), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "DoubTech/Loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        v = comfy.sd.VAE(ckpt_path=vae.download(vae_name))
        return (v,)


class DTLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (lora.list(), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "DoubTech/Loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = lora.download("lora_name")
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)
        return (model_lora, clip_lora)


class DTCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clip.list(), ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/Loaders"

    def load_clip(self, clip_name):
        clip_path = clip.download(clip_name)
        c = comfy.sd.load_clip(ckpt_path=clip_path, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (c,)


class DTCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clipVision.list(), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/Loaders"

    def load_clip(self, clip_name):
        clip_path = clipVision.download(clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)


class DTStyleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "style_model_name": (style.list(), )}}

    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "load_style_model"

    CATEGORY = "DoubTech/Loaders"

    def load_style_model(self, style_model_name):
        style_model_path = style.download(style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)
        return (style_model,)


class DTGLIGENLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"gligen_name": (gligen.list(),)}}

    RETURN_TYPES = ("GLIGEN",)
    FUNCTION = "load_gligen"

    CATEGORY = "DoubTech/Loaders"

    def load_gligen(self, gligen_name):
        gligen_path = gligen.download(gligen_name)
        g = comfy.sd.load_gligen(gligen_path)
        return (g,)


class DTControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (controlNet.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/Loaders"

    def load_controlnet(self, control_net_name):
        controlnet_path = controlNet.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path)
        return (controlnet,)


class DTDiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (controlNetDiff.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/Loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = controlNetDiff.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path, model)
        return (controlnet,)


class DTunCLIPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (unclipCheckpoint.list(), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = unclipCheckpoint.download(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

class DTCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (configs.list(), ),
                              "ckpt_name": (checkpoints.list(), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/advanced/loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        config_path = configs.download(config_name)
        ckpt_path = checkpoints.download(ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

class DTDiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/advanced/loaders"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return comfy.diffusers_load.load_diffusers(model_path, fp16=comfy.model_management.should_use_fp16(), output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))


class DTLoadLatent:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")]
        return {"required": {"latent": [sorted(files), ]}, }

    CATEGORY = "DoubTech/_for_testing"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load"

    def load(self, latent):
        latent_path = folder_paths.get_annotated_filepath(latent)
        latent = safetensors.torch.load_file(latent_path, device="cpu")
        samples = {"samples": latent["latent_tensor"].float()}
        return (samples, )

    @classmethod
    def IS_CHANGED(s, latent):
        image_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if not folder_paths.exists_annotated_filepath(latent):
            return "Invalid latent file: {}".format(latent)
        return True



class DTLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://www.doubtech.ai/img/doubtech.ai-qrcode.png"
                })},
                }

    CATEGORY = "DoubTech/image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        i = Image.open(image)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    @classmethod
    def IS_CHANGED(s, image):
        #image_path = folder_paths.get_annotated_filepath(image)
        #m = hashlib.sha256()
        #with open(image_path, 'rb') as f:
        #    m.update(f.read())
        #return m.digest().hex()
        return True

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not image:
            return "Invalid image file: {}".format(image)

        return True

class DTLoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://www.doubtech.ai/img/doubtech.ai-qrcode.png"
                }),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "DoubTech/mask"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        # Load the image from a url
        i = Image.open(image)
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask,)

    @classmethod
    def IS_CHANGED(s, image, channel):
        """image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()"""
        return True

    @classmethod
    def VALIDATE_INPUTS(s, image, channel):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        if channel not in s._color_channels:
            return "Invalid color channel: {}".format(channel)

        return True

class DTUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (upscalers.list(), ), }}

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "DoubTech/Loaders"

    def load_model(self, model_name):
        model_path = upscalers.download(model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTSubmitImage": SubmitImage,
    "DTCheckpointLoaderSimple": DTNodeCheckpointLoader,
    "DTVAELoader": DTVAELoader,
    "DTLoraLoader": DTLoraLoader,
    "DTCLIPLoader": DTCLIPLoader,
    "DTControlNetLoader": DTControlNetLoader,
    "DTDiffControlNetLoader": DTDiffControlNetLoader,
    "DTStyleModelLoader": DTStyleModelLoader,
    "DTCLIPVisionLoader": DTCLIPVisionLoader,
    "DTunCLIPCheckpointLoader": DTunCLIPCheckpointLoader,
    "DTGLIGENLoader": DTGLIGENLoader,
    "DTCheckpointLoader": DTCheckpointLoader,
    "DTDiffusersLoader": DTDiffusersLoader,
    "DTLoadLatent": DTLoadLatent,
    "DTLoadImage": DTLoadImage,
    "DTLoadImageMask": DTLoadImageMask,
    "DTUpscaleModelLoader": DTUpscaleModelLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DTSubmitImage": "Submit Image",
    "DTCheckpointLoader": "Load Checkpoint (With Config - Online)",
    "DTCheckpointLoaderSimple": "Load Checkpoint (Online)",
    "DTVAELoader": "Load VAE (Online)",
    "DTLoraLoader": "Load LoRA (Online)",
    "DTCLIPLoader": "Load CLIP (Online)",
    "DTControlNetLoader": "Load ControlNet Model (Online)",
    "DTDiffControlNetLoader": "Load ControlNet Model (diff) (Online)",
    "DTStyleModelLoader": "Load Style Model (Online)",
    "DTCLIPVisionLoader": "Load CLIP Vision (Online)",
    "DTUpscaleModelLoader": "Load Upscale Model (Online)",
    "DTPreviewImage": "Preview Image (Online)",
    "DTLoadImage": "Load Image (Online)",
}
