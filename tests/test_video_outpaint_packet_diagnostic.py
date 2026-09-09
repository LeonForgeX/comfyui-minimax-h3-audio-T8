from tools.diagnose_outpaint_video_packets import byte_difference_summary


def test_byte_diagnostic_distinguishes_bit_flips_and_length(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"abc\x00\xffz")
    b.write_bytes(b"abc\x01\x00")
    value = byte_difference_summary(a, b)
    assert value["different_shared_bytes"] == 2
    assert value["different_shared_bits"] == 9
    assert value["length_difference"] == 1
    assert value["first_differences"][0] == {"offset": 3, "left": 0, "right": 1, "xor": 1}


def test_byte_diagnostic_chunk_boundary_and_identical(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"a" * 1024**2 + b"\x02")
    b.write_bytes(b"a" * 1024**2 + b"\x03")
    assert byte_difference_summary(a, b)["first_differences"][0]["offset"] == 1024**2
    assert byte_difference_summary(a, a)["different_shared_bits"] == 0
