"""Watermark payload schema definition.

Payload encodes: artist_id, album_id, optional manifest_id pointer,
and integrity bits (CRC) into a fixed-length bit vector.
"""

import struct
import zlib


PAYLOAD_BITS = 32  # total payload size in bits


def encode_payload(artist_id: int, album_id: int, manifest_id: int = 0) -> bytes:
    """Encode artist/album/manifest IDs into a watermark payload.

    Layout (32 bits):
        artist_id:   10 bits (0-1023)
        album_id:     8 bits (0-255)
        manifest_id:  6 bits (0-63)
        crc8:         8 bits (integrity check)

    Returns:
        4 bytes representing the payload.
    """
    assert 0 <= artist_id < 1024, f"artist_id must be 0-1023, got {artist_id}"
    assert 0 <= album_id < 256, f"album_id must be 0-255, got {album_id}"
    assert 0 <= manifest_id < 64, f"manifest_id must be 0-63, got {manifest_id}"

    packed = (artist_id << 22) | (album_id << 14) | (manifest_id << 8)
    # CRC8 of the upper 24 bits
    data_bytes = struct.pack(">I", packed)[:-1]  # upper 3 bytes
    crc = zlib.crc32(data_bytes) & 0xFF
    packed |= crc

    return struct.pack(">I", packed)


def decode_payload(payload_bytes: bytes) -> dict:
    """Decode a watermark payload and verify integrity.

    Returns:
        dict with keys: artist_id, album_id, manifest_id, crc_valid
    """
    value = struct.unpack(">I", payload_bytes)[0]

    artist_id = (value >> 22) & 0x3FF
    album_id = (value >> 14) & 0xFF
    manifest_id = (value >> 8) & 0x3F
    crc_received = value & 0xFF

    # Verify CRC
    packed_no_crc = value & 0xFFFFFF00
    data_bytes = struct.pack(">I", packed_no_crc)[:-1]
    crc_computed = zlib.crc32(data_bytes) & 0xFF

    return {
        "artist_id": artist_id,
        "album_id": album_id,
        "manifest_id": manifest_id,
        "crc_valid": crc_received == crc_computed,
    }
