mkdir -p build
cd build
cmake ..
make

ln -sf ../../../cactus/ggml-llama.metallib default.metallib
echo "\n--- Running Core API VLM Example ---"
./cactus_vlm_core

echo "\n--- Waiting 5 seconds before running FFI example ---"
sleep 5

echo "\n--- Running FFI VLM Example ---"
./cactus_vlm_ffi