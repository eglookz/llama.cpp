First, build and compile the project with the following commands (make sure you have `make` and `cmake` installeds):
`cmake --build build --config Release -j 8`

Download the model file `tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf` from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF and place it in the `build/bin/models` directory after the successfull compilation.
Now you can run the command to test the model:
`./build/bin/llama-cli -m ./build/bin/models/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf -p "Explain what is quantum computing in simple terms." -t 4 --temp 0.7 --top-p 0.9 --n-predict 128`


./build/bin/llama-cli -m ./build/bin/models/mistral-7b-instruct-v0.2.Q8_0.gguf -p "What is the goal of Large Language Model?" -t 4 -c 512 -n 64 --temp 0.7 --top-p 0.9 --no-warmup -ub 256 -b 512

watch -n 0.5 '
  echo "=== MEMORY ==="
  free -h
  echo ""
  echo "=== LLAMA PROCESS ==="
  ps aux | grep llama-cli | grep -v grep | awk "{print \$6/1024 \" MB\"}"
'
