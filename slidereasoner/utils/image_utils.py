import math
from PIL import ImageColor, Image, ImageDraw, ImageFont
from typing import Literal, Tuple, Annotated, Optional, Union, List, Any, Dict, Sequence

import openslide




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


def infer_level0_mpp(
    slide: openslide.OpenSlide,
    override_level0_mpp: Optional[float] = None,
) -> float:
    """Infer level-0 mpp from OpenSlide properties."""

    if override_level0_mpp is not None:
        if override_level0_mpp <= 0:
            raise ValueError(
                f"override_level0_mpp must be positive, got {override_level0_mpp}",
            )
        return float(override_level0_mpp)

    properties = slide.properties
    candidates = (
        "openslide.mpp-x",
        "aperio.MPP",
        "hamamatsu.XResolution",
    )

    for key in candidates:
        value = properties.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            if key == "hamamatsu.XResolution":
                # Hamamatsu often stores pixels per mm; convert to um/px.
                return 1000.0 / parsed
            return parsed

    raise ValueError(
        "Failed to infer level0 mpp from slide properties. "
        "Please provide `level0_mpp` explicitly when starting the session.",
    )

def validate_relative_bbox(bbox_2d: Sequence[int]) -> Tuple[int, int, int, int]:
    """Validate a 0-1000 relative bbox."""

    if len(bbox_2d) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {bbox_2d}")

    rel_x1, rel_y1, rel_x2, rel_y2 = [int(v) for v in bbox_2d]
    values = (rel_x1, rel_y1, rel_x2, rel_y2)
    if any(v < 0 or v > 1000 for v in values):
        raise ValueError(
            f"bbox_2d values must all be within [0, 1000], got {bbox_2d}",
        )
    if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
        raise ValueError(
            f"bbox_2d must satisfy x2>x1 and y2>y1, got {bbox_2d}",
        )
    return rel_x1, rel_y1, rel_x2, rel_y2


def validate_patch_pixels_for_wsi(
    bbox_2d: Sequence[int],
    width: int,
    height: int,
    observation_index: int,
    source_mpp: float,
    target_mpp: float,
    factor: int = 32,
) -> None:
    """Validate image size constraints for model-side vision ingestion."""

    min_pixels = IMAGE_MIN_TOKEN_NUM * factor * factor
    max_pixels = IMAGE_MAX_TOKEN_NUM * factor * factor
    pixels = width * height
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d

    if min(width, height) <= 0:
        raise ValueError(
            f"Invalid patch size {width}x{height} for bbox_2d={bbox_2d}",
        )

    if max(width, height) / min(width, height) > MAX_RATIO:
        raise ValueError(
            "WSI_PATCH_INVALID_ASPECT_RATIO: "
            f"bbox_2d={list(bbox_2d)}; size={width}x{height}; "
            f"max_ratio={MAX_RATIO}. Action: adjust bbox to reduce the aspect ratio.",
        )

    if pixels > max_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_LARGE: "
            f"observation_index={observation_index}; "
            f"source_mpp={source_mpp:.6f}; target_mpp={target_mpp:.6f}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"patch_wh={width}x{height}; patch_pixels={pixels}; "
            f"max_pixels={max_pixels}. "
            "Action: shrink bbox (reduce area) and retry.",
        )

    if pixels < min_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_SMALL: "
            f"observation_index={observation_index}; "
            f"source_mpp={source_mpp:.6f}; target_mpp={target_mpp:.6f}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"patch_wh={width}x{height}; patch_pixels={pixels}; "
            f"min_pixels={min_pixels}. "
            "Action: expand bbox (increase area) and retry.",
        )


SnapMode = Literal["expand", "shrink", "nearest"]


def snap_to_multiple(value: int, multiple: int, mode: SnapMode) -> int:
    """
    将整数对齐到指定倍数。

    例如 multiple=32 时：
    - expand:  510 -> 512
    - shrink:  510 -> 480
    - nearest: 510 -> 512

    Args:
        value:
            原始整数值。
        multiple:
            目标倍数，例如 32。
        mode:
            对齐策略：
            - "expand": 向上取整
            - "shrink": 向下取整
            - "nearest": 选最近的倍数

    Returns:
        对齐后的值。
    """
    if multiple <= 1:
        return max(1, value)

    if mode == "expand":
        return max(multiple, math.ceil(value / multiple) * multiple)

    if mode == "shrink":
        return max(multiple, math.floor(value / multiple) * multiple)

    down = max(multiple, math.floor(value / multiple) * multiple)
    up = max(multiple, math.ceil(value / multiple) * multiple)
    return down if abs(value - down) <= abs(up - value) else up


def snap_down_to_multiple(value: int, multiple: int) -> int:
    """
    将整数向下对齐到指定倍数。

    这个函数主要用于“空间不足时”的兜底逻辑：
    当期望输出尺寸对应的 native ROI 已经放不进 slide 内，
    且又不想 padding，这时只能把输出尺寸缩小到“当前可行的最大倍数”。

    Args:
        value:
            原始整数值。
        multiple:
            目标倍数，例如 32。

    Returns:
        不大于 value 的最大 multiple 倍数。
    """
    if multiple <= 1:
        return max(1, value)
    return max(multiple, math.floor(value / multiple) * multiple)



def fit_interval_to_bounds(center: float, size: int, limit: int) -> Tuple[int, int]:
    """
    Place a 1D interval inside valid bounds while preserving its size when possible.

    The interval is represented as [start, start + size), and the function tries
    to keep it centered around `center`. If the interval would exceed the valid
    range [0, limit), it is shifted back into bounds instead of being resized,
    unless the requested size is larger than the full valid range.

    Args:
        center:
            Desired center of the interval.
        size:
            Desired interval length.
        limit:
            Upper bound of the valid coordinate range.

    Returns:
        A tuple (start, adjusted_size), where:
        - start is a valid integer start position
        - adjusted_size is clipped to `limit` if necessary

    Raises:
        ValueError:
            If `limit` is not positive.
    """
    if limit <= 0:
        raise ValueError(f"Invalid limit={limit}; it must be positive.")

    # If the requested interval is larger than the available range,
    # clip it to the maximum allowed size.
    # size = min(size, limit)

    if size > limit:
        raise ValueError(f"Invalid size={size}; it must be smaller than limit={limit}.")

    start = int(round(center - size / 2.0))

    # Shift the interval so that it remains fully inside the valid range.
    start = max(0, min(start, limit - size))

    return start, size


def get_roi_at_native_resolution(
    slide: openslide.OpenSlide,
    source_roi: Tuple[int, int, int, int],
    source_mpp: float,
    source_native_x: int,
    source_native_y: int,
    source_native_w: int,
    source_native_h: int,
    target_mpp: float,
    native_mpp: float,
    min_pixels: int = 32,
    patch_multiple: int = 32,
    snap_mode: SnapMode = "expand",
) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
    """
    从 WSI 的 native 分辨率（即 OpenSlide level 0）读取 ROI，并输出目标尺度图像。

    ---------------------------
    一、这个函数解决什么问题
    ---------------------------
    在多尺度 WSI 浏览中，我们通常已经有一个“父 observation”，
    它本身对应 WSI 上的一块区域。现在模型在这个父 observation 上
    选中了一个子框 source_roi，希望继续放大查看。

    这个函数的作用就是：
    1. 把 source_roi 从“父 observation 的像素坐标系”映射回
       “WSI 的 native 坐标系”
    2. 在 native 坐标系中得到真实的 ROI
    3. 用目标 mpp 计算应输出的图像大小
    4. 必要时把输出宽高对齐到 patch_multiple 的倍数
    5. 通过轻微外扩 / 缩小 native ROI，尽量让最终输出尺寸天然匹配模型输入
    6. 永远从 native 分辨率读取，再 resize 一次，避免依赖 OpenSlide 的 level 选择逻辑

    ---------------------------
    二、为什么统一从 native 读取
    ---------------------------
    这里我们故意不使用 slide.level_downsamples 去找“最接近的层”。

    好处：
    - 逻辑更简单、更统一
    - 所有 ROI 都来自同一个原生坐标系，方便调试和可解释性追踪
    - 更容易精确控制“最终输出尺寸”和“目标放大尺度”的关系

    代价：
    - 在低倍率、大视野时，直接从 native 读取会更耗内存和 I/O

    ---------------------------
    三、几个关键坐标系
    ---------------------------
    1. 父 observation 坐标系
       - source_roi = (x, y, w, h)
       - 单位是“父 observation 图像上的像素”
       - 例如模型在当前图上框出一个矩形

    2. native 坐标系
       - native == OpenSlide level 0
       - 是 WSI 的原生最高分辨率坐标系
       - source_native_x/y/w/h 表示“父 observation 对应到 WSI 上的真实区域”

    3. 输出图像坐标系
       - 最终返回给模型的 patch 图像像素坐标系
       - 宽高由 target_mpp 和对齐规则共同决定

    ---------------------------
    四、关于 patch_multiple
    ---------------------------
    一些视觉模型会要求输入宽高在预处理时被对齐到固定倍数。
    如果这一步发生在模型内部，就可能引入额外 resize。

    为了减少这种“不可控的二次缩放”，这里可以直接在 ROI 提取阶段：
    - 先计算理论输出宽高
    - 再对齐到 patch_multiple
    - 然后反推应该读取多大的 native ROI

    这样模型拿到的图更接近我们想要的目标尺度。

    Args:
        slide:
            OpenSlide 打开的 WSI 对象。

        source_roi:
            当前要提取的子 ROI，在父 observation 图像坐标系下表示为：
            (x, y, width, height)

        source_mpp:
            父 observation 的有效 mpp。
            它表示父 observation 图像中 1 个像素，对应真实切片中多少微米。

        source_native_x, source_native_y, source_native_w, source_native_h:
            父 observation 在 native 坐标系中的真实位置和大小。
            也就是说，父 observation 对应 WSI 上的哪一块区域。

        target_mpp:
            目标输出尺度。
            数值越小，表示放大倍率越高。
            如果 target_mpp 小于 native_mpp，说明你请求的分辨率高于原生分辨率，
            这种情况不可能真正获得更多细节，因此会自动 clamp 到 native_mpp。

        native_mpp:
            WSI 原生分辨率的 mpp，即 OpenSlide level 0 的 mpp。

        min_pixels:
            输出图像任意一边的最小像素数，防止 ROI 太小导致输出图像过小。

        patch_multiple:
            输出宽高需要对齐到的倍数，例如 32。
            如果不想做这一步，可以传 1。

        snap_mode:
            输出尺寸对齐策略：
            - "expand": 优先外扩 ROI，保留更多上下文
            - "shrink": 优先缩小 ROI，减少背景
            - "nearest": 选最接近的那个

    Returns:
        image:
            最终输出的 RGB patch。

        native_roi:
            最终实际读取的 ROI，在 native 坐标系中的位置：
            (x, y, width, height)

        effective_mpp:
            实际输出图像对应的 mpp。
            当 target_mpp 小于 native_mpp 时，会退化为 native_mpp。
    """
    # ---------------------------
    # 1) 基本参数合法性检查
    # ---------------------------
    if source_mpp <= 0 or target_mpp <= 0 or native_mpp <= 0:
        raise ValueError(f"source_mpp={source_mpp}, target_mpp={target_mpp}, and native_mpp={native_mpp} must all be positive.")

    slide_w, slide_h = slide.dimensions

    src_x, src_y, src_w, src_h = source_roi
    if src_x < 0 or src_y < 0 or src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source_roi={source_roi}; all dimensions must be positive.")

    if source_native_x < 0 or source_native_y < 0 or source_native_w <= 0 or source_native_h <= 0:
        raise ValueError(
            f"source_native_x {source_native_x}/source_native_y {source_native_y} must be non-negative and "
            f"source_native_w {source_native_w}/source_native_h {source_native_h} must be positive."
        )

    # 父 observation 在整张切片中的右下角
    parent_x2 = source_native_x + source_native_w
    parent_y2 = source_native_y + source_native_h

    # 父 observation 本身必须是 slide 内的合法区域
    if parent_x2 > slide_w or parent_y2 > slide_h:
        raise ValueError(
            f"Parent native extent exceeds slide bounds: "
            f"parent=({source_native_x}, {source_native_y}, {source_native_w}, {source_native_h}), "
            f"slide_size={slide.dimensions}"
        )

    # ---------------------------
    # 2) 把子框从父 observation 坐标映射到 native 坐标
    # ---------------------------
    # source_mpp / native_mpp 表示：
    # 父 observation 图像中的 1 个像素，相当于 native 坐标系中的多少个像素
    #
    # 举例：
    # 如果父 observation 是 1.0 mpp，而 native 是 0.25 mpp，
    # 那么父图的 1 个像素 ~= native 的 4 个像素
    scale_to_native = source_mpp / native_mpp

    # 计算子 ROI 在 native 坐标系中的左上角
    native_x = source_native_x + math.floor(src_x * scale_to_native)
    native_y = source_native_y + math.floor(src_y * scale_to_native)

    # 计算子 ROI 在 native 坐标系中的宽高
    native_w = max(1, math.floor(src_w * scale_to_native))
    native_h = max(1, math.floor(src_h * scale_to_native))

    # 检查映射后的 ROI 是否还落在父 observation 内
    if native_x < source_native_x or native_y < source_native_y:
        raise ValueError("Mapped ROI starts outside parent extent.")

    if native_x + native_w > parent_x2 or native_y + native_h > parent_y2:
        raise ValueError(
            "Mapped native ROI exceeds parent observation extent: "
            f"roi=({native_x}, {native_y}, {native_w}, {native_h}), "
            f"parent=({source_native_x}, {source_native_y}, {source_native_w}, {source_native_h})"
        )

    # ---------------------------
    # 3) 决定实际输出尺度 effective_mpp
    # ---------------------------
    # 如果 target_mpp 比 native_mpp 还小，意味着你想“看得比原图还清楚”，
    # 这在物理上做不到，所以只能退回 native_mpp
    effective_mpp = max(target_mpp, native_mpp)

    # ---------------------------
    # 4) 根据目标尺度计算理论输出尺寸
    # ---------------------------
    # native_w * native_mpp / effective_mpp
    # 表示：native 宽度在目标尺度下应该映射成多少像素
    #
    # 例如：
    # native_w = 1024, native_mpp = 0.25, effective_mpp = 0.5
    # 则输出宽度约为 1024 * 0.25 / 0.5 = 512
    raw_out_w = max(min_pixels, round(native_w * native_mpp / effective_mpp))
    raw_out_h = max(min_pixels, round(native_h * native_mpp / effective_mpp))

    # ---------------------------
    # 5) 把输出尺寸对齐到模型友好的倍数
    # ---------------------------
    # 例如 Qwen 这类模型，如果内部会把输入长宽调整到 32 的倍数，
    # 那我们宁可自己先控制这一步，而不是让模型在黑盒里再做一次 resize。
    out_w = snap_to_multiple(raw_out_w, patch_multiple, snap_mode)
    out_h = snap_to_multiple(raw_out_h, patch_multiple, snap_mode)

    # ---------------------------
    # 6) 反推应读取多大的 native ROI
    # ---------------------------
    # 这里的关键思想是：
    # 如果输出宽高变了，为了保持“目标倍率 / mpp”一致，
    # 对应的 native ROI 也应该同步变化。
    #
    # 例如：
    # 原本理论输出宽度是 510，但为了对齐 32 倍数变成了 512，
    # 那么 native ROI 也应该稍微调整，让 512 这个输出宽度在当前 effective_mpp 下成立。
    aligned_native_w = max(1, round(out_w * effective_mpp / native_mpp))
    aligned_native_h = max(1, round(out_h * effective_mpp / native_mpp))

    # ---------------------------
    # 7) 尽量保持中心不变，重新定位调整后的 native ROI
    # ---------------------------
    # 这里默认是“围绕原中心对称地外扩/缩小”，
    # 这样比直接只改右边界或下边界更自然，也更不容易让 ROI 漂移。
    center_x = native_x + native_w / 2.0
    center_y = native_y + native_h / 2.0

    aligned_native_x, aligned_native_w = fit_interval_to_bounds(center_x, aligned_native_w, slide_w)
    aligned_native_y, aligned_native_h = fit_interval_to_bounds(center_y, aligned_native_h, slide_h)

    native_roi = (
        aligned_native_x,
        aligned_native_y,
        aligned_native_w,
        aligned_native_h,
    )

    # ---------------------------
    # 8) 从 native 分辨率直接读取 ROI
    # ---------------------------
    # 这里强制使用 OpenSlide level 0，也就是 native resolution。
    # 不再依赖 level_downsamples 去挑“最近的层”。
    crop = slide.read_region(
        (aligned_native_x, aligned_native_y),
        0,
        (aligned_native_w, aligned_native_h),
    ).convert("RGB")

    # ---------------------------
    # 9) resize 到最终输出尺寸
    # ---------------------------
    # 如果读取到的 native patch 大小和目标输出大小不同，
    # 就进行一次显式 resize。这样图像缩放行为是可控的。
    if crop.size != (out_w, out_h):
        crop = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)

    # 返回：
    # - 最终 patch 图像
    # - 实际读取的 native ROI
    # - 实际输出使用的 mpp
    return crop, native_roi, effective_mpp

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
