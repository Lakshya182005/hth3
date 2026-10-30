"""Unit tests for image validation and safe decoding."""

import pytest
from PIL import Image

from campus_hunt_cv.config import ImageLimitsConfig
from campus_hunt_cv.exceptions import ImageValidationError
from campus_hunt_cv.image_validation import validate_and_decode_image
from tests.conftest import create_test_image_bytes


def test_valid_jpeg_decoding():
    """Valid JPEG bytes should be decoded into RGB PIL Image."""
    data = create_test_image_bytes(width=120, height=80, format="JPEG")
    img = validate_and_decode_image(data)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (120, 80)


def test_valid_png_decoding():
    """Valid PNG bytes should be decoded into RGB PIL Image."""
    data = create_test_image_bytes(width=64, height=64, format="PNG")
    img = validate_and_decode_image(data)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_rgba_converted_to_rgb():
    """RGBA images must be cleanly composited onto RGB without alpha artifacts."""
    data = create_test_image_bytes(width=50, height=50, mode="RGBA")
    img = validate_and_decode_image(data)
    assert img.mode == "RGB"
    assert img.size == (50, 50)


def test_grayscale_converted_to_rgb():
    """Grayscale 'L' images must be converted to 3-channel RGB for neural network input."""
    data = create_test_image_bytes(width=40, height=40, mode="L", color=128)
    img = validate_and_decode_image(data)
    assert img.mode == "RGB"
    assert img.size == (40, 40)


def test_empty_image_bytes():
    """Empty byte buffers must raise ImageValidationError."""
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(b"")
    assert exc_info.value.reason_code == "ERROR_EMPTY_IMAGE_BYTES"


def test_corrupted_garbage_bytes():
    """Random unidentifiable bytes must raise ImageValidationError."""
    garbage = b"\x00\x01\x02random_non_image_garbage_bytes_12345"
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(garbage)
    assert exc_info.value.reason_code in ("ERROR_IMAGE_UNIDENTIFIED", "ERROR_IMAGE_CORRUPTED")


def test_truncated_image_bytes():
    """Truncated image stream must fail verification."""
    data = create_test_image_bytes(width=100, height=100, format="JPEG")
    truncated = data[:len(data) // 4]
    with pytest.raises(ImageValidationError):
        validate_and_decode_image(truncated)


def test_oversized_payload_bytes():
    """Payloads exceeding max_image_bytes must be rejected."""
    limits = ImageLimitsConfig(max_image_bytes=1024)
    data = create_test_image_bytes(width=800, height=800, format="PNG")
    assert len(data) > 1024

    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(data, limits=limits)
    assert exc_info.value.reason_code == "ERROR_IMAGE_TOO_LARGE_BYTES"


def test_dimension_below_minimum():
    """Images with width or height below min_image_dimension must be rejected."""
    limits = ImageLimitsConfig(min_image_dimension=50)
    data = create_test_image_bytes(width=30, height=60, format="JPEG")

    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(data, limits=limits)
    assert exc_info.value.reason_code == "ERROR_IMAGE_TOO_SMALL_DIMENSION"


def test_decompression_bomb_guard():
    """Images exceeding max_image_pixels must be rejected to prevent DoS attacks."""
    limits = ImageLimitsConfig(max_image_pixels=10_000)
    data = create_test_image_bytes(width=200, height=200, format="PNG")  # 40,000 pixels > 10,000

    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(data, limits=limits)
    assert exc_info.value.reason_code == "ERROR_IMAGE_DECOMPRESSION_BOMB"


def test_unsupported_format_rejected():
    """Formats outside allowed_formats must be rejected."""
    limits = ImageLimitsConfig(allowed_formats=("PNG",))
    data = create_test_image_bytes(width=60, height=60, format="JPEG")

    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_decode_image(data, limits=limits)
    assert exc_info.value.reason_code == "ERROR_IMAGE_UNSUPPORTED_FORMAT"
