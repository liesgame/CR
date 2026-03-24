import math
from PIL import ImageColor, Image, ImageDraw, ImageFont
from typing import Literal, Tuple, Annotated, Optional, Union, List, Any, Dict, Sequence


MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8


# mulit_print
IPYNB_MAX_IMG_DIM = 400


# image reasize and crop
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")
      
def smart_resize(height: int, width: int, factor: int = 32) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels =  (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels =  (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    return h_bar, w_bar

def validate_MinMax_pixels(bbox, w_bar, h_bar, image_idx, origin_mag, target_mag, max_pixels, min_pixels):
    
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox

    pixels = w_bar * h_bar
    coord_note = (
        "Qwen3-VL bbox coords are relative on a 0-1000 scale "
        "Convert to pixels: abs_px = rel/1000 * (img_w,img_h)."
    )    
    if pixels > max_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_LARGE: "
            f"image_idx={image_idx}; "
            f"origin_wsi_magnification=x{origin_mag}; "
            f"target_wsi_magnification=x{target_mag}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"corresponding_image_wh={w_bar}x{h_bar}; "
            f"patch_pixels={pixels} > max_pixels={max_pixels}; "
            f"coord_note={coord_note} "
            "Action: shrink bbox (reduce area) and retry with same image_idx & target_wsi_magnification."
        )

    elif pixels < min_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_SMALL: "
            f"image_idx={image_idx}; "
            f"origin_wsi_magnification=x{origin_mag}; "
            f"target_wsi_magnification=x{target_mag}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"corresponding_image_wh={w_bar}x{h_bar}; "
            f"patch_pixels={pixels} < min_pixels={min_pixels}; "
            f"coord_note={coord_note} "
            "Action: expand_bbox (increase area) and retry with same image_idx & target_wsi_magnification."
        )
    
def validate_MinMax_pixels_test(bbox, w_bar, h_bar, max_pixels, min_pixels):

    
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox

    pixels = w_bar * h_bar
    coord_note = (
        "Qwen3-VL bbox coords are relative on a 0-1000 scale "
        "Convert to pixels: abs_px = rel/1000 * (img_w,img_h)."
    )    
    if pixels > max_pixels:
        raise ValueError(
            "BBOX_PATCH_TOO_LARGE: "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"corresponding_image_wh={w_bar}x{h_bar}; "
            f"patch_pixels={pixels} > max_pixels={max_pixels}; "
            f"coord_note={coord_note} "
            "Action: shrink bbox (reduce area)"
        )

    elif pixels < min_pixels:
        raise ValueError(
            "BBOX_PATCH_TOO_SMALL: "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"corresponding_image_wh={w_bar}x{h_bar}; "
            f"patch_pixels={pixels} < min_pixels={min_pixels}; "
            f"coord_note={coord_note} "
            "Action: expand_bbox (increase area)"
        )
    
def maybe_resize_bbox(left, top, right, bottom, img_width, img_height, factor=32):
    """Resize bbox to ensure it's valid"""
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    height = bottom - top
    width = right - left
    if height < factor or width < factor:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = factor / min(height, width)
        new_half_height = math.ceil(height * ratio * 0.5)
        new_half_width = math.ceil(width * ratio * 0.5)
        new_left = math.floor(center_x - new_half_width)
        new_right = math.ceil(center_x + new_half_width)
        new_top = math.floor(center_y - new_half_height)
        new_bottom = math.ceil(center_y + new_half_height)

        # Ensure the resized bbox is within image bounds
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_width, new_right)
        new_bottom = min(img_height, new_bottom)

        new_height = new_bottom - new_top
        new_width = new_right - new_left

        if new_height >= factor and new_width >= factor:
            return [new_left, new_top, new_right, new_bottom]
        else:
            raise ValueError(f"new_height: {new_height} or new_width: {new_width} < 32, need to increase the bbox area ")
    return [left, top, right, bottom]


## multi-print
def in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False
    

def resize_keep_ratio_max_side(img, max_side: int = IPYNB_MAX_IMG_DIM):
    """Keep aspect ratio; ensure longest side <= max_side."""
    if not max_side or max_side <= 0:
        return img

    try:
        resample = __import__("PIL.Image").Image.Resampling.LANCZOS  # Pillow >= 9
    except Exception:
        from PIL import Image  # type: ignore
        resample = getattr(Image, "LANCZOS", 1)

    w, h = img.size
    if max(w, h) <= max_side:
        return img

    img = img.copy()
    img.thumbnail((max_side, max_side), resample=resample)
    return img

def display_image_from_source(
    source: Dict[str, Any],
    print_image: bool = True,
    save_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Returns:
        - Jupyter: None (already displayed)
        - Terminal: returns file path if saved, or None if only printed URL
    """
    if not print_image:
        return None

    src_type = source.get("type")
    if src_type not in ("url", "base64"):
        raise ValueError(f"image source type must be 'url' or 'base64', got: {src_type}")

    is_jup = in_jupyter()

    # Lazy imports (optional deps)
    PIL_OK = True
    try:
        from PIL import Image  # type: ignore
    except Exception:
        PIL_OK = False

    if src_type == "url":
        url = source.get("url")
        if not url:
            return None

        if is_jup and PIL_OK:
            try:
                import requests  # type: ignore
                from IPython.display import display  # type: ignore

                # img = Image.open(requests.get(url, stream=True, timeout=20).raw)
                img = Image.open(url)
                img = resize_keep_ratio_max_side(img)
                display(img)
                return None
            except Exception as e:
                print(f"[IMAGE] failed to display from url: {url} ({e})")
                print(f"[IMAGE] {url}")
                return None

        # terminal / no PIL
        print(f"[IMAGE] {url}")
        return None
    else:
        raise NotImplementedError(f"src_type : {src_type} is not implemented")
