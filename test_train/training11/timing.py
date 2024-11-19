import time
import zstandard as zstd
import torch
from processing import *
import io
import random

which = 0

if which == 0:
    # save regular file
    a = torch.randint(0, 2, (400, 1000), dtype=torch.bool)
    start_time = time.time()
    torch.save(a, "a.pt")

    a = torch.randint(0, 2, (400, 1000), dtype=torch.int8)
    start_time = time.time()
    torch.save(a, "b.pt")

if which == 1:
    # save compressed file
    a = torch.randint(0, 3, (400, 1000)).to(torch.int8)
    start_time = time.time()
    buffer = io.BytesIO()
    torch.save(a, buffer)
    
    serialized_tensor = buffer.getvalue()
    
    compressor = zstd.ZstdCompressor()
    compressed_data = compressor.compress(serialized_tensor)
    
    with open("a.pt.zst", 'wb') as f:
        f.write(compressed_data)

if which == 2:
    # load regular file

    start_time = time.time()
    a = torch.load("a.pt")

if which == 3:
    # load compressed file

    with open("a.pt.zst", 'rb') as f:
        compressed_data = f.read()
    
    decompressor = zstd.ZstdDecompressor()
    decompressed_data = decompressor.decompress(compressed_data)
    
    buffer = io.BytesIO(decompressed_data)
    a = torch.load(buffer)

print(time.time() - start_time)