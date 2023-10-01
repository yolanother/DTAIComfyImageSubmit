import uuid

import boto3

import json
import urllib
import io
import base64

from PIL.PngImagePlugin import PngInfo
import numpy as np
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


class SubmitImage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
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
            s3_client.upload_fileobj(
                img_bytes,
                config.s3_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/png'
                }
            )
            print(f"Image uploaded to S3 bucket: {config.s3_bucket}, key: {s3_key}")

            # Generate the URL of the uploaded image
            s3_url = f"https://{config.s3_bucket}.s3.amazonaws.com/{s3_key}"
            return s3_url
        except Exception as e:
            print("Error uploading image to S3:", e)
            return None


    def upload(self, image, prompt_text="", tags="", title="", alt="", caption="", set="", private="", prompt=None, extra_pnginfo=None):
        print("uploading image...")

        print("Extras: ", extra_pnginfo)

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
        # if user is set in extra_pnginfo, use that, otherwise use the user from the global variables

        user = ""
        if "user" in extra_pnginfo:
            user = urllib.parse.quote(variables.apply(extra_pnginfo["user"]))

        job = ""
        if "job" in extra_pnginfo:
            job = urllib.parse.quote(variables.apply(f'{extra_pnginfo["job"]}'))


        # Create a post request to submit the image as the post body to the backend
        uri = "https://api.aiart.doubtech.com/comfyui/submit?key={}&tags={}&title={}&alt={}&set={}&prompt={}&caption={}&private={}&job={}&user={}".format(
            config.apikey,
            tags,
            title,
            alt,
            setName,
            prompt_text,
            caption,
            private,
            job,
            user)

        print(f"Submitting {prompt_text} with data:\n{prompt}")

        images = image
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

class DTSimpleSubmitImage(SubmitImage):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "DoubTech/Image"

    OUTPUT_NODE = True

    def submit(self, image, prompt=None, extra_pnginfo=None):
        return super().upload(
            image,
            prompt_text=variables.state["prompt"],
            tags=variables.state["tags"],
            title=variables.state["title"],
            alt=variables.state["alt"],
            caption=variables.state["caption"],
            set=variables.state["set"],
            private=variables.state["private"],
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTSubmitImage": SubmitImage,
    "DTSimpleSubmitImage": DTSimpleSubmitImage,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DTSubmitImage": "Submit Image (Parameters)",
    "DTSimpleSubmitImage": "Submit Image",
}
