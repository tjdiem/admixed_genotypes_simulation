import zstandard as zstd
import time
import os
import gzip 
import shutil
import lz4.frame

file = "saved_inputs/X_file6_chunk156000.pt"

cctx = zstd.ZstdCompressor(level=3)

start_time = time.time()

# out_file = file + ".gz"
# with open(file, "rb") as f_in:
#     with gzip.open(out_file, "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

out_file = file + ".zst"
with open(file, "rb") as f_in:
    with open(out_file, "wb") as f_out:
        cctx.copy_stream(f_in, f_out)

# out_file = file + ".lz4"
# with open(file, "rb") as f_in:
#     data = f_in.read()
#     compressed_data = lz4.frame.compress(data)

#     with open(out_file, "wb") as f_out:
#         f_out.write(compressed_data)

print("time: ", time.time() - start_time)

print("size: ", os.path.getsize(out_file) / os.path.getsize(file))


