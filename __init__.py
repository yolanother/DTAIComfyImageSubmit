import base64
import io
import urllib
from urllib import request
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage

from custom_nodes.DTAIComfyImageSubmit import config


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

    CATEGORY = "image"

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SubmitImage": SubmitImage
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SubmitImage": "Submit Image to DoubTech.ai"
}
