import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def adjust_brightness_contrast(pil: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    """Brightness/contrast adjustment. brightness/contrast in ~[0.5, 1.5]."""
    img = ImageEnhance.Brightness(pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

def add_gaussian_noise(pil: Image.Image, sigma: float = 10.0) -> Image.Image:
    """Additive Gaussian noise with std=sigma in [0..255] scale."""
    arr = np.array(pil).astype(np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def gaussian_blur(pil: Image.Image, radius: float = 1.5) -> Image.Image:
    return pil.filter(ImageFilter.GaussianBlur(radius))

def jpeg_compress(pil: Image.Image, quality: int = 30) -> Image.Image:
    from io import BytesIO
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def occlude_rectangle(pil: Image.Image, box: tuple[int,int,int,int], color=(0,0,0)) -> Image.Image:
    """Draw a solid rectangle occluder at (left, top, right, bottom)."""
    from PIL import ImageDraw
    img = pil.copy()
    d = ImageDraw.Draw(img)
    d.rectangle(box, fill=color)
    return img

def occlude_eyes(pil: Image.Image, frac_height: float = 0.2) -> Image.Image:
    w, h = pil.size
    top = int(h * 0.30)
    height = int(h * frac_height)
    return occlude_rectangle(pil, (0, top, w, top + height))

def occlude_mouth(pil: Image.Image, frac_height: float = 0.25) -> Image.Image:
    w, h = pil.size
    top = int(h * 0.65)
    height = int(h * frac_height)
    return occlude_rectangle(pil, (0, top, w, min(h, top + height)))


