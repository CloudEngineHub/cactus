# !/bin/bash -e

yarn --cwd react

# fetch llama.cpp at a fixed commit instead of using a submodule
LLAMA_CPP_COMMIT=18b663d8e4ef352a9a15ff15d695fc3258801d60
LLAMA_CPP_DIR=llama.cpp

# clean up any previous copy
rm -rf "$LLAMA_CPP_DIR"

# shallow‑clone and checkout the wanted commit
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
(cd "$LLAMA_CPP_DIR" && git fetch --depth 1 origin $LLAMA_CPP_COMMIT && git checkout $LLAMA_CPP_COMMIT)

cp ./llama.cpp/include/llama.h ./cactus/llama.h
cp ./llama.cpp/include/llama-cpp.h ./cactus/llama-cpp.h

cp ./llama.cpp/ggml/include/ggml.h ./cactus/ggml.h
cp ./llama.cpp/ggml/include/ggml-alloc.h ./cactus/ggml-alloc.h
cp ./llama.cpp/ggml/include/ggml-backend.h ./cactus/ggml-backend.h
cp ./llama.cpp/ggml/include/ggml-cpu.h ./cactus/ggml-cpu.h
cp ./llama.cpp/ggml/include/ggml-cpp.h ./cactus/ggml-cpp.h
cp ./llama.cpp/ggml/include/ggml-opt.h ./cactus/ggml-opt.h
cp ./llama.cpp/ggml/include/ggml-metal.h ./cactus/ggml-metal.h
cp ./llama.cpp/ggml/include/gguf.h ./cactus/gguf.h

cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal.m ./cactus/ggml-metal.m
cp ./llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h ./cactus/ggml-metal-impl.h

cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c ./cactus/ggml-cpu.c
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp ./cactus/ggml-cpu.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h ./cactus/ggml-cpu-impl.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.h ./cactus/ggml-cpu-aarch64.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ./cactus/ggml-cpu-aarch64.cpp
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.h ./cactus/ggml-cpu-quants.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c ./cactus/ggml-cpu-quants.c
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-traits.h ./cactus/ggml-cpu-traits.h
cp ./llama.cpp/ggml/src/ggml-cpu/ggml-cpu-traits.cpp ./cactus/ggml-cpu-traits.cpp

cp -r ./llama.cpp/ggml/src/ggml-cpu/amx ./cactus/

cp ./llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.h ./cactus/sgemm.h
cp ./llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp ./cactus/sgemm.cpp

cp ./llama.cpp/ggml/src/ggml.c ./cactus/ggml.c
cp ./llama.cpp/ggml/src/ggml-impl.h ./cactus/ggml-impl.h
cp ./llama.cpp/ggml/src/ggml-alloc.c ./cactus/ggml-alloc.c
cp ./llama.cpp/ggml/src/ggml-backend.cpp ./cactus/ggml-backend.cpp
cp ./llama.cpp/ggml/src/ggml-backend-impl.h ./cactus/ggml-backend-impl.h
cp ./llama.cpp/ggml/src/ggml-backend-reg.cpp ./cactus/ggml-backend-reg.cpp
cp ./llama.cpp/ggml/src/ggml-common.h ./cactus/ggml-common.h
cp ./llama.cpp/ggml/src/ggml-opt.cpp ./cactus/ggml-opt.cpp
cp ./llama.cpp/ggml/src/ggml-quants.h ./cactus/ggml-quants.h
cp ./llama.cpp/ggml/src/ggml-quants.c ./cactus/ggml-quants.c
cp ./llama.cpp/ggml/src/ggml-threading.cpp ./cactus/ggml-threading.cpp
cp ./llama.cpp/ggml/src/ggml-threading.h ./cactus/ggml-threading.h
cp ./llama.cpp/ggml/src/gguf.cpp ./cactus/gguf.cpp

cp ./llama.cpp/src/llama.cpp ./cactus/llama.cpp
cp ./llama.cpp/src/llama-chat.h ./cactus/llama-chat.h
cp ./llama.cpp/src/llama-chat.cpp ./cactus/llama-chat.cpp
cp ./llama.cpp/src/llama-context.h ./cactus/llama-context.h
cp ./llama.cpp/src/llama-context.cpp ./cactus/llama-context.cpp
cp ./llama.cpp/src/llama-mmap.h ./cactus/llama-mmap.h
cp ./llama.cpp/src/llama-mmap.cpp ./cactus/llama-mmap.cpp
cp ./llama.cpp/src/llama-kv-cache.h ./cactus/llama-kv-cache.h
cp ./llama.cpp/src/llama-kv-cache.cpp ./cactus/llama-kv-cache.cpp
cp ./llama.cpp/src/llama-model-loader.h ./cactus/llama-model-loader.h
cp ./llama.cpp/src/llama-model-loader.cpp ./cactus/llama-model-loader.cpp
cp ./llama.cpp/src/llama-model.h ./cactus/llama-model.h
cp ./llama.cpp/src/llama-model.cpp ./cactus/llama-model.cpp
cp ./llama.cpp/src/llama-adapter.h ./cactus/llama-adapter.h
cp ./llama.cpp/src/llama-adapter.cpp ./cactus/llama-adapter.cpp
cp ./llama.cpp/src/llama-arch.h ./cactus/llama-arch.h
cp ./llama.cpp/src/llama-arch.cpp ./cactus/llama-arch.cpp
cp ./llama.cpp/src/llama-batch.h ./cactus/llama-batch.h
cp ./llama.cpp/src/llama-batch.cpp ./cactus/llama-batch.cpp
cp ./llama.cpp/src/llama-cparams.h ./cactus/llama-cparams.h
cp ./llama.cpp/src/llama-cparams.cpp ./cactus/llama-cparams.cpp
cp ./llama.cpp/src/llama-hparams.h ./cactus/llama-hparams.h
cp ./llama.cpp/src/llama-hparams.cpp ./cactus/llama-hparams.cpp
cp ./llama.cpp/src/llama-impl.h ./cactus/llama-impl.h
cp ./llama.cpp/src/llama-impl.cpp ./cactus/llama-impl.cpp

cp ./llama.cpp/src/llama-vocab.h ./cactus/llama-vocab.h
cp ./llama.cpp/src/llama-vocab.cpp ./cactus/llama-vocab.cpp
cp ./llama.cpp/src/llama-grammar.h ./cactus/llama-grammar.h
cp ./llama.cpp/src/llama-grammar.cpp ./cactus/llama-grammar.cpp
cp ./llama.cpp/src/llama-sampling.h ./cactus/llama-sampling.h
cp ./llama.cpp/src/llama-sampling.cpp ./cactus/llama-sampling.cpp

cp ./llama.cpp/src/unicode.h ./cactus/unicode.h
cp ./llama.cpp/src/unicode.cpp ./cactus/unicode.cpp
cp ./llama.cpp/src/unicode-data.h ./cactus/unicode-data.h
cp ./llama.cpp/src/unicode-data.cpp ./cactus/unicode-data.cpp

cp ./llama.cpp/src/llama-graph.h ./cactus/llama-graph.h
cp ./llama.cpp/src/llama-graph.cpp ./cactus/llama-graph.cpp
cp ./llama.cpp/src/llama-io.h ./cactus/llama-io.h
cp ./llama.cpp/src/llama-io.cpp ./cactus/llama-io.cpp
cp ./llama.cpp/src/llama-memory.h ./cactus/llama-memory.h
cp ./llama.cpp/src/llama-memory.cpp ./cactus/llama-memory.cpp

cp ./llama.cpp/common/log.h ./cactus/log.h
cp ./llama.cpp/common/log.cpp ./cactus/log.cpp
cp ./llama.cpp/common/common.h ./cactus/common.h
cp ./llama.cpp/common/common.cpp ./cactus/common.cpp
cp ./llama.cpp/common/sampling.h ./cactus/sampling.h
cp ./llama.cpp/common/sampling.cpp ./cactus/sampling.cpp
cp ./llama.cpp/common/json-schema-to-grammar.h ./cactus/json-schema-to-grammar.h
cp ./llama.cpp/common/json-schema-to-grammar.cpp ./cactus/json-schema-to-grammar.cpp
cp ./llama.cpp/common/json.hpp ./cactus/json.hpp

cp ./llama.cpp/common/chat.h ./cactus/chat.h
cp ./llama.cpp/common/chat.cpp ./cactus/chat.cpp

cp ./llama.cpp/common/minja/minja.hpp ./cactus/minja/minja.hpp
cp ./llama.cpp/common/minja/chat-template.hpp ./cactus/minja/chat-template.hpp

List of files to process
files_add_lm_prefix=(
  "./cactus/llama-impl.h"
  "./cactus/llama-impl.cpp"
  "./cactus/llama-vocab.h"
  "./cactus/llama-vocab.cpp"
  "./cactus/llama-grammar.h"
  "./cactus/llama-grammar.cpp"
  "./cactus/llama-sampling.h"
  "./cactus/llama-sampling.cpp"
  "./cactus/llama-adapter.h"
  "./cactus/llama-adapter.cpp"
  "./cactus/llama-arch.h"
  "./cactus/llama-arch.cpp"
  "./cactus/llama-batch.h"
  "./cactus/llama-batch.cpp"
  "./cactus/llama-chat.h"
  "./cactus/llama-chat.cpp"
  "./cactus/llama-context.h"
  "./cactus/llama-context.cpp"
  "./cactus/llama-kv-cache.h"
  "./cactus/llama-kv-cache.cpp"
  "./cactus/llama-model-loader.h"
  "./cactus/llama-model-loader.cpp"
  "./cactus/llama-model.h"
  "./cactus/llama-model.cpp"
  "./cactus/llama-mmap.h"
  "./cactus/llama-mmap.cpp"
  "./cactus/llama-hparams.h"
  "./cactus/llama-hparams.cpp"
  "./cactus/llama-cparams.h"
  "./cactus/llama-cparams.cpp"
  "./cactus/llama-graph.h"
  "./cactus/llama-graph.cpp"
  "./cactus/llama-io.h"
  "./cactus/llama-io.cpp"
  "./cactus/llama-memory.h"
  "./cactus/llama-memory.cpp"
  "./cactus/log.h"
  "./cactus/log.cpp"
  "./cactus/llama.h"
  "./cactus/llama.cpp"
  "./cactus/sampling.cpp"
  "./cactus/sgemm.h"
  "./cactus/sgemm.cpp"
  "./cactus/common.h"
  "./cactus/common.cpp"
  "./cactus/json-schema-to-grammar.h"
  "./cactus/chat.cpp"
  "./cactus/ggml-common.h"
  "./cactus/ggml.h"
  "./cactus/ggml.c"
  "./cactus/gguf.h"
  "./cactus/gguf.cpp"
  "./cactus/ggml-impl.h"
  "./cactus/ggml-cpp.h"
  "./cactus/ggml-opt.h"
  "./cactus/ggml-opt.cpp"
  "./cactus/ggml-metal.h"
  "./cactus/ggml-metal.m"
  "./cactus/ggml-metal-impl.h"
  "./cactus/ggml-quants.h"
  "./cactus/ggml-quants.c"
  "./cactus/ggml-alloc.h"
  "./cactus/ggml-alloc.c"
  "./cactus/ggml-backend.h"
  "./cactus/ggml-backend.cpp"
  "./cactus/ggml-backend-impl.h"
  "./cactus/ggml-backend-reg.cpp"
  "./cactus/ggml-cpu-impl.h"
  "./cactus/ggml-cpu.h"
  "./cactus/ggml-cpu.c"
  "./cactus/ggml-cpu.cpp"
  "./cactus/ggml-cpu-aarch64.h"
  "./cactus/ggml-cpu-aarch64.cpp"
  "./cactus/ggml-cpu-quants.h"
  "./cactus/ggml-cpu-quants.c"
  "./cactus/ggml-cpu-traits.h"
  "./cactus/ggml-cpu-traits.cpp"
  "./cactus/ggml-threading.h"
  "./cactus/ggml-threading.cpp"
  "./cactus/amx/amx.h"
  "./cactus/amx/amx.cpp"
  "./cactus/amx/mmq.h"
  "./cactus/amx/mmq.cpp"
  "./cactus/amx/common.h"
)

# Loop through each file and run the sed commands
OS=$(uname)
for file in "${files_add_lm_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/GGML_/LM_GGML_/g' $file
    sed -i '' 's/ggml_/lm_ggml_/g' $file
    sed -i '' 's/GGUF_/LM_GGUF_/g' $file
    sed -i '' 's/gguf_/lm_gguf_/g' $file
    sed -i '' 's/GGMLMetalClass/LMGGMLMetalClass/g' $file
  else
    sed -i 's/GGML_/LM_GGML_/g' $file
    sed -i 's/ggml_/lm_ggml_/g' $file
    sed -i 's/GGUF_/LM_GGUF_/g' $file
    sed -i 's/gguf_/lm_gguf_/g' $file
    sed -i 's/GGMLMetalClass/LMGGMLMetalClass/g' $file
  fi
done

files_iq_add_lm_prefix=(
  "./cactus/ggml-quants.h"
  "./cactus/ggml-quants.c"
  "./cactus/ggml.c"
)

for file in "${files_iq_add_lm_prefix[@]}"; do
  # Add prefix to avoid redefinition with other libraries using ggml like whisper.rn
  if [ "$OS" = "Darwin" ]; then
    sed -i '' 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i '' 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i '' 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i '' 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  else
    sed -i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' $file
    sed -i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' $file
    sed -i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' $file
    sed -i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' $file
  fi
done

echo "Replacement completed successfully!"

# yarn --cwd example

# Apply patch
patch -p0 -d ./cactus < ./cactus/patches/common.h.patch
patch -p0 -d ./cactus < ./cactus/patches/common.cpp.patch
patch -p0 -d ./cactus < ./cactus/patches/chat.h.patch
patch -p0 -d ./cactus < ./cactus/patches/chat.cpp.patch
patch -p0 -d ./cactus < ./cactus/patches/log.cpp.patch
patch -p0 -d ./cactus < ./cactus/patches/ggml-metal.m.patch
patch -p0 -d ./cactus < ./cactus/patches/ggml.c.patch
patch -p0 -d ./cactus < ./cactus/patches/ggml-quants.c.patch
patch -p0 -d ./cactus < ./cactus/patches/llama-mmap.cpp.patch
rm -rf ./cactus/*.orig

if [ "$OS" = "Darwin" ]; then
  # Build metallib (~2.6MB)
  cd llama.cpp/ggml/src/ggml-metal

  # Create a symbolic link to ggml-common.h in the current directory
  ln -sf ../ggml-common.h .

  xcrun --sdk iphoneos metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphoneos metallib ggml-metal.air   -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib ../../../../cactus/ggml-llama.metallib

  xcrun --sdk iphonesimulator metal -c ggml-metal.metal -o ggml-metal.air -DGGML_METAL_USE_BF16=1
  xcrun --sdk iphonesimulator metallib ggml-metal.air   -o ggml-llama.metallib
  rm ggml-metal.air
  mv ./ggml-llama.metallib ../../../../cactus/ggml-llama-sim.metallib

  # Remove the symbolic link
  rm ggml-common.h

  cd -
  
fi

# after we're finished, remove the temporary clone to keep the tree clean
rm -rf "$LLAMA_CPP_DIR"
