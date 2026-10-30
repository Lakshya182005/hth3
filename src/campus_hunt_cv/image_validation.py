"""Image validation and safe decoding module.

Ensures that participant-submitted bytes are sanitized, within size limits,
protected against decompression bombs, and correctly converted to RGB
before being forwarded to the neural network pipeline.
"""

from __future__ import annotations

import io
from typing import Optional
from PIL import Image, ImageOps, UnidentifiedImageError

from campus_hunt_cv.config import ImageLimitsConfig
from campus_hunt_cv.exceptions import ImageValidationError


def validate_and_decode_image(
    image_bytes: bytes,
    limits: Optional[ImageLimitsConfig] = None,
) -> Image.Image:
    """Safely validate, decode, and normalize raw image bytes into a PIL RGB Image.

    Performs defense-in-depth validation:
    1. Checks byte buffer length against maximum allowed size.
    2. Enforces decompression bomb thresholds (Pillow MAX_IMAGE_PIXELS).
    3. Verifies file header and integrity without executing untrusted codecs.
    4. Enforces format restrictions (e.g. JPEG, PNG, WEBP).
    5. Validates minimum/maximum pixel dimensions.
    6. Normalizes color channels to standard RGB (converting RGBA, grayscale, CMYK, or palette).
    7. Applies EXIF orientation to ensure upright orientation matching participant's view.

    Args:
        image_bytes: Raw binary image payload.
        limits: Optional custom limits configuration. If None, default limits apply.

    Returns:
        A loaded, upright PIL Image in 'RGB' mode.

    Raises:
        ImageValidationError: If the payload is empty, corrupted, oversized, or unsupported.
    """
    if limits is None:
        limits = ImageLimitsConfig()

    # 1. Byte length check
    if not image_bytes or len(image_bytes) == 0:
        raise ImageValidationError(
            "Image byte payload is empty.",
            reason_code="ERROR_EMPTY_IMAGE_BYTES",
        )

    byte_size = len(image_bytes)
    if byte_size > limits.max_image_bytes:
        raise ImageValidationError(
            f"Image payload size ({byte_size} bytes) exceeds maximum permitted limit "
            f"({limits.max_image_bytes} bytes).",
            reason_code="ERROR_IMAGE_TOO_LARGE_BYTES",
        )

    # Configure PIL safety limit
    Image.MAX_IMAGE_PIXELS = limits.max_image_pixels

    # 2. Header and format verification
    try:
        bio = io.BytesIO(image_bytes)
        with Image.open(bio) as probe_img:
            format_name = (probe_img.format or "").upper()
            allowed = tuple(fmt.upper() for fmt in limits.allowed_formats)
            if format_name not in allowed:
                raise ImageValidationError(
                    f"Unsupported image format: '{format_name}'. Allowed formats: {allowed}.",
                    reason_code="ERROR_IMAGE_UNSUPPORTED_FORMAT",
                )

            # Check dimensions before decoding full buffer
            width, height = probe_img.size
            if width < limits.min_image_dimension or height < limits.min_image_dimension:
                raise ImageValidationError(
                    f"Image dimensions ({width}x{height}) are below minimum required "
                    f"({limits.min_image_dimension}x{limits.min_image_dimension}).",
                    reason_code="ERROR_IMAGE_TOO_SMALL_DIMENSION",
                )

            total_pixels = width * height
            if total_pixels > limits.max_image_pixels:
                raise ImageValidationError(
                    f"Image total pixels ({total_pixels}) exceeds decompression bomb limit "
                    f"({limits.max_image_pixels}).",
                    reason_code="ERROR_IMAGE_DECOMPRESSION_BOMB",
                )

            # Run verify() to check header integrity (note: verify invalidates image state)
            probe_img.verify()

    except Image.DecompressionBombError as e:
        raise ImageValidationError(
            f"Image triggered decompression bomb guard: {str(e)}",
            reason_code="ERROR_IMAGE_DECOMPRESSION_BOMB",
        ) from e
    except UnidentifiedImageError as e:
        raise ImageValidationError(
            "Uploaded file is not a recognized image or is severely corrupted.",
            reason_code="ERROR_IMAGE_UNIDENTIFIED",
        ) from e
    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(
            f"Failed image verification check: {str(e)}",
            reason_code="ERROR_IMAGE_CORRUPTED",
        ) from e

    # 3. Full decode and color normalization
    try:
        bio = io.BytesIO(image_bytes)
        img = Image.open(bio)
        img.load()  # Force reading all image data into memory

        # Handle EXIF orientation (e.g. mobile phone camera orientation tags)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            # If EXIF transposition fails, proceed with the original image orientation
            pass

        # Convert to RGB mode
        if img.mode == "RGB":
            return img
        elif img.mode == "RGBA":
            # Composite onto neutral white background to avoid dark transparent artifacts
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        elif img.mode in ("P", "LA"):
            return img.convert("RGBA").convert("RGB")
        else:
            return img.convert("RGB")

    except Image.DecompressionBombError as e:
        raise ImageValidationError(
            f"Image triggered decompression bomb guard during decoding: {str(e)}",
            reason_code="ERROR_IMAGE_DECOMPRESSION_BOMB",
        ) from e
    except Exception as e:
        raise ImageValidationError(
            f"Image decoding failed: {str(e)}",
            reason_code="ERROR_IMAGE_DECODE_FAILED",
        ) from e

