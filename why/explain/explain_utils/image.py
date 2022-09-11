import io
import random
import copy
import string

from PIL import Image
import cv2
import numpy as np
import matplotlib.cm as mpl_color_map


def generate_random_name(N):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )


def visualize(map, image_size, channel):
    """
    Temporary visualization function
    """
    map = 255 * (map - map.min()) / (map.max() - map.min())
    map = cv2.resize(map, image_size)
    map = np.stack([map] * channel, -1)
    map = map.astype(np.uint8)
    return map


def array_handler(dl_array):
    if "torch" in str(type(dl_array)):
        return dl_array.numpy()
    else:
        return dl_array


def overlay_heatmap_on_original_image(
    original_image,
    heatmap,
    filename=None,
    image_size=None,
    alpha=0.5,
    colormap_name="RdYlBu",
    return_bytes=False,
):
    """
    Heatmap overlay
    """
    # Check if image is PIL or NumPy
    is_pil_or_np = True if "PIL" in str(type(original_image)) else False

    if image_size:
        if is_pil_or_np:
            original_image = original_image.resize(image_size)
        else:
            original_image = cv2.resize(original_image, image_size)
            original_image = Image.fromarray(original_image)

        heatmap = array_handler(heatmap)
        heatmap = np.array(Image.fromarray(heatmap).resize(image_size))
    else:
        image_size = (
            original_image.size
            if is_pil_or_np
            else [i for i in original_image.shape if i > 5]
        )

    color_map = mpl_color_map.get_cmap(colormap_name)

    # Change alpha channel in colormap to make sure original image is displayed
    heatmap_colored = color_map(heatmap[:, :, 0], alpha=alpha)
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap = copy.copy(heatmap_colored)

    heatmap = Image.fromarray((heatmap).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", original_image.size)
    heatmap_on_image = Image.alpha_composite(
        heatmap_on_image, original_image.convert("RGBA")
    )
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    if return_bytes:
        img_byte_arr = io.BytesIO()
        heatmap_on_image.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()
    else:
        if filename:
            is_extension = filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            )
            filename = filename if is_extension else filename + ".png"
            heatmap_on_image.save(filename)
            return filename
        else:
            filename = generate_random_name(10) + ".png"
            heatmap_on_image.save(filename)
            return filename


def resize_heatmap_wo_original_image(
    heatmap,
    filename=None,
    image_size=None,
    alpha=0.5,
    colormap_name="RdYlBu",
    return_bytes=False,
):
    """
    Heatmap overlay
    """

    if image_size:
        heatmap = np.array(Image.fromarray(heatmap).resize(image_size))

    color_map = mpl_color_map.get_cmap(colormap_name)

    # Change alpha channel in colormap to make sure original image is displayed
    heatmap_colored = color_map(heatmap[:, :, 0], alpha=alpha)
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap = copy.copy(heatmap_colored)

    heatmap = Image.fromarray((heatmap).astype(np.uint8))

    if return_bytes:
        img_byte_arr = io.BytesIO()
        heatmap.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()
    else:
        if filename:
            is_extension = filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            )
            filename = filename if is_extension else filename + ".png"
            heatmap.save(filename)
            return filename
        else:
            return heatmap


def create_polygon(mask):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    polygons = []
    for object in contours:
        for point in object:
            polygons.append({"x": int(point[0][0]), "y": int(point[0][1])})
        break
    return polygons
