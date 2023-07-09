import torch

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

        # Use hashlib to create the md5 hash of the key
        hash_object = hashlib.md5(key.encode())
        md5_hash = hash_object.hexdigest()

        # Combine self.path with md5_hash to get the filepath
        return folder_paths.get_full_path(self.path, md5_hash)

    def download(self, key):
        if key not in self.data:
            raise KeyError(f"Key {key} not found in {self.uri}")

        filename = self.filename(key)

        # If the file doesn't exist at that path, download and save it
        if not os.path.exists(filename):
            urllib.request.urlretrieve(self.data[key], filename)

        return filename


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
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
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
        }

    RETURN_TYPES = ()
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "upload"

    OUTPUT_NODE = True

    CATEGORY = "DoubTech/image"

    def upload(self, image, prompt, tags, title, alt, caption, set, private):
        print("uploading image...")

        # uriencode the parameters
        tags = urllib.parse.quote(tags)
        title = urllib.parse.quote(title)
        alt = urllib.parse.quote(alt)
        setName = urllib.parse.quote(set)
        prompt = urllib.parse.quote(prompt)
        caption = urllib.parse.quote(caption)

        # Create a post request to submit the image as the post body to the backend
        uri = "https://api.aiart.doubtech.com/comfyui/submit?key={}&tags={}&title={}&alt={}&set={}&prompt={}&caption={}&private={}".format(config.apikey, tags, title, alt, setName, prompt, caption, private)

        num_images = image.size(0)
        #iterate over the images
        for i in range(num_images):
            print("There are " + str(num_images) + " images in the batch")
            img = image[i]

            # Convert the image to a png
            png = ToPILImage()(img.permute(2, 0, 1))
            img_bytes = io.BytesIO()
            png.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # Encode the image bytes to base64
            encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

            print("Submission uri: " + uri)
            # Submit the image using a POST request
            with request.urlopen(request.Request(uri, data=encoded.encode('utf-8'), method='POST')) as f:
                print(f.read().decode('utf-8'))

            return ()


class NodeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (checkpoints.list(), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        pass


class VAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (vae.list(), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "DoubTech/loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        v = comfy.sd.VAE(ckpt_path=vae.download(vae_name))
        return (v,)


class LoraLoader:
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

    CATEGORY = "DoubTech/loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = lora.download("lora_name")
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)
        return (model_lora, clip_lora)


class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clip.list(), ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/loaders"

    def load_clip(self, clip_name):
        clip_path = clip.download(clip_name)
        c = comfy.sd.load_clip(ckpt_path=clip_path, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (c,)


class CLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clipVision.list(), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/loaders"

    def load_clip(self, clip_name):
        clip_path = clipVision.download(clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)


class StyleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "style_model_name": (style.list(), )}}

    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "load_style_model"

    CATEGORY = "DoubTech/loaders"

    def load_style_model(self, style_model_name):
        style_model_path = style.download(style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)
        return (style_model,)


class GLIGENLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"gligen_name": (gligen.list(),)}}

    RETURN_TYPES = ("GLIGEN",)
    FUNCTION = "load_gligen"

    CATEGORY = "DoubTech/loaders"

    def load_gligen(self, gligen_name):
        gligen_path = gligen.download(gligen_name)
        g = comfy.sd.load_gligen(gligen_path)
        return (g,)


class ControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (controlNet.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/loaders"

    def load_controlnet(self, control_net_name):
        controlnet_path = controlNet.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path)
        return (controlnet,)


class DiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (controlNetDiff.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = controlNetDiff.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path, model)
        return (controlnet,)


class unCLIPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (unclipCheckpoint.list(), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = unclipCheckpoint.download(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

class CheckpointLoader:
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

class DiffusersLoader:
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


class LoadLatent:
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



class LoadImage:
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

class LoadImageMask:
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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTSubmitImage": SubmitImage,
    "DTCheckpointLoaderSimple": NodeCheckpointLoader,
    "DTVAELoader": VAELoader,
    "DTLoraLoader": LoraLoader,
    "DTCLIPLoader": CLIPLoader,
    "DTControlNetLoader": ControlNetLoader,
    "DTDiffControlNetLoader": DiffControlNetLoader,
    "DTStyleModelLoader": StyleModelLoader,
    "DTCLIPVisionLoader": CLIPVisionLoader,
    "DTunCLIPCheckpointLoader": unCLIPCheckpointLoader,
    "DTGLIGENLoader": GLIGENLoader,
    "DTCheckpointLoader": CheckpointLoader,
    "DTDiffusersLoader": DiffusersLoader,
    "DTLoadLatent": LoadLatent,
    "DTLoadImage": LoadImage,
    "DTLoadImageMask": LoadImageMask,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DTCheckpointLoader": "Load Checkpoint (With Config)",
    "DTCheckpointLoaderSimple": "Load Checkpoint",
    "DTVAELoader": "Load VAE",
    "DTLoraLoader": "Load LoRA",
    "DTCLIPLoader": "Load CLIP",
    "DTControlNetLoader": "Load ControlNet Model",
    "DTDiffControlNetLoader": "Load ControlNet Model (diff)",
    "DTStyleModelLoader": "Load Style Model",
    "DTCLIPVisionLoader": "Load CLIP Vision",
    "DTUpscaleModelLoader": "Load Upscale Model",
    "DTPreviewImage": "Preview Image",
    "DTLoadImage": "Load Image",
}
