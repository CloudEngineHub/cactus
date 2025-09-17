#!/usr/bin/env python3
import numpy as np
import struct
import sys
import json
import argparse
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Please install required packages: pip install torch transformers")
    sys.exit(1)

def save_tensor_with_header(tensor, output_path, precision='FP32', transpose=False, stats_tracker=None, args=None):
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)
    
    original_data = data.copy()
    mean_val = np.mean(original_data)
    std_val = np.std(original_data)
    min_val = np.min(original_data)
    max_val = np.max(original_data)
    
    
    if precision == 'INT8':
        filename = output_path.name
        if any(x in filename for x in ['norm']):
            precision = 'FP16'
    
    if precision == 'INT8':
        qmin, qmax = -128, 127
        standard_scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
        
        standard_zero_point = qmax - max_val / standard_scale
        standard_zero_point_clipped = np.clip(np.round(standard_zero_point), qmin, qmax)
        test_quantized = np.clip(np.round(original_data / standard_scale + standard_zero_point_clipped), qmin, qmax)
        test_saturation = np.sum(np.abs(test_quantized) >= 127) / original_data.size
        
        saturation_threshold = args.saturation_threshold if args else 0.01
        if test_saturation > saturation_threshold:
            outlier_percentile = args.outlier_percentile if args else 0.01
            lower_percentile = np.percentile(original_data, outlier_percentile)
            upper_percentile = np.percentile(original_data, 100 - outlier_percentile)
            
            mean_val = np.mean(original_data)
            std_val = np.std(original_data)
            sigma_multiplier = args.sigma_multiplier if args else 3.5
            three_sigma_min = mean_val - sigma_multiplier * std_val
            three_sigma_max = mean_val + sigma_multiplier * std_val
            
            clipped_min = max(min_val, min(lower_percentile, three_sigma_min))
            clipped_max = min(max_val, max(upper_percentile, three_sigma_max))
            
            range_threshold = args.range_threshold if args else 0.5
            if (clipped_max - clipped_min) < range_threshold * (max_val - min_val):
                clipped_min = min_val
                clipped_max = max_val
        else:
            clipped_min = min_val
            clipped_max = max_val
        
        # Symmetric quantization: use maximum absolute value for optimal range
        abs_max = max(abs(clipped_min), abs(clipped_max))
        scale = abs_max / 127.0 if abs_max != 0 else 1.0
        
        quantized_data = np.clip(np.round(original_data / scale), qmin, qmax).astype(np.int8)
        
        dequantized_data = quantized_data.astype(np.float32) * scale
        mse_error = np.mean((original_data - dequantized_data) ** 2)
        snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')
        
        original_flat = original_data.flatten()
        dequantized_flat = dequantized_data.flatten()
        cos_sim = np.dot(original_flat, dequantized_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequantized_flat))
        
        snr_threshold = args.snr_threshold if args else 30.0
        if snr_db < snr_threshold:
            precision = 'FP16'
            data = data.astype(np.float16)
            scale = 1.0
            if stats_tracker:
                stats_tracker['low_snr_fallbacks'] = stats_tracker.get('low_snr_fallbacks', 0) + 1
        else:
            saturated_values = np.sum(np.abs(quantized_data) == 127)
            saturation_percent = (saturated_values / quantized_data.size) * 100
            data = quantized_data
            
            if stats_tracker:
                stats_tracker['quantized_tensors'] += 1
                stats_tracker['quantized_parameters'] += original_data.size
                stats_tracker['mse_values'].append(mse_error)
                stats_tracker['snr_values'].append(snr_db)
                stats_tracker['cos_sim_values'].append(cos_sim)
                saturation_warning_threshold = args.saturation_warning_threshold if args else 0.1
                if saturation_percent > saturation_warning_threshold:
                    stats_tracker['saturation_warnings'] += 1
    elif precision == 'FP16':
        data = data.astype(np.float16)
        scale = 1.0
    else:
        data = data.astype(np.float32)
        scale = 1.0
    
    if stats_tracker:
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += original_data.size
    
    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        shape = [shape[1], shape[0]]
    
    data = data.flatten()
    
    print(f"Saving {output_path.name}: {precision} {shape}")
    
    with open(output_path, 'wb') as f:
        ndim = len(shape)
        f.write(struct.pack('<I', ndim))
        
        for dim in shape:
            f.write(struct.pack('<Q', dim))
        
        if precision == 'INT8':
            prec_val = 0
        elif precision == 'FP16':
            prec_val = 1
        else: 
            prec_val = 2
        f.write(struct.pack('<I', prec_val))
        
        if precision == 'INT8':
            element_size = 1
        elif precision == 'FP16':
            element_size = 2
        else: 
            element_size = 4
        byte_size = data.size * element_size
        f.write(struct.pack('<Q', byte_size))
        
        if precision == 'INT8':
            f.write(struct.pack('<f', scale))
            
        f.write(data.tobytes())
    
    if precision == 'INT8':
        scale_path = output_path.with_suffix('.scale')
        with open(scale_path, 'w') as f:
            f.write(f"{scale:.10f}\n")

def convert_hf_model_weights(model, output_dir, precision='INT8', args=None):
    
    quantization_stats = {
        'total_tensors': 0,
        'quantized_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }
    
    state_dict = model.state_dict()
    config = model.config
    
    
    tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
    
    model_config = {
        'vocab_size': getattr(config, 'vocab_size', 0),
        'hidden_dim': getattr(config, 'hidden_size', 0),
        'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 0)),
        'attention_heads': getattr(config, 'num_attention_heads', 0),
        'attention_kv_heads': getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 0)),
        'ffn_intermediate_dim': getattr(config, 'intermediate_size', 0),
        'context_length': getattr(config, 'max_position_embeddings', getattr(config, 'max_sequence_length', 0)),
        'rope_theta': getattr(config, 'rope_theta', 10000.0),
        'attention_head_dim': getattr(config, 'head_dim', getattr(config, 'hidden_size', 0) // getattr(config, 'num_attention_heads', 1)),
        'tie_word_embeddings': tie_word_embeddings
    }
    
    embed_names = ['model.embed_tokens.weight', 'embed_tokens.weight', 'embeddings.weight', 'transformer.wte.weight']
    embedding_found = False
    embedding_tensor = None
    for name in embed_names:
        if name in state_dict:
            embedding_tensor = state_dict[name]
            save_tensor_with_header(embedding_tensor, output_dir / "token_embeddings.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args)
            embedding_found = True
            break
    
    if not embedding_found:
        pass
    
    if not tie_word_embeddings:
        output_names = ['lm_head.weight', 'output.weight', 'transformer.lm_head.weight']
        for name in output_names:
            if name in state_dict:
                tensor = state_dict[name]
                save_tensor_with_header(tensor, output_dir / "output_weight.weights", precision, transpose=False, stats_tracker=quantization_stats, args=args)
                break
    
    output_norm_names = ['model.norm.weight', 'norm.weight', 'final_layernorm.weight', 'transformer.ln_f.weight']
    for name in output_norm_names:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(tensor, output_dir / "output_norm.weights", precision, stats_tracker=quantization_stats, args=args)
            break
    
    num_layers = model_config['num_layers']
    for i in range(num_layers):
        
        layer_prefixes = [f'model.layers.{i}.', f'layers.{i}.', f'transformer.h.{i}.']
        
        layer_prefix = None
        for prefix in layer_prefixes:
            if any(key.startswith(prefix) for key in state_dict.keys()):
                layer_prefix = prefix
                break
        
        if not layer_prefix:
            continue
        
        weight_patterns = [
            (['self_attn.q_proj.weight', 'attn.q_proj.weight', 'attn.c_attn.weight'], precision, f'layer_{i}_attn_q.weights', False),
            (['self_attn.k_proj.weight', 'attn.k_proj.weight'], precision, f'layer_{i}_attn_k.weights', False),
            (['self_attn.v_proj.weight', 'attn.v_proj.weight'], precision, f'layer_{i}_attn_v.weights', False),
            (['self_attn.o_proj.weight', 'attn.o_proj.weight', 'attn.c_proj.weight'], precision, f'layer_{i}_attn_output.weights', False),
            (['input_layernorm.weight', 'ln_1.weight'], precision, f'layer_{i}_input_norm.weights', False),
            (['self_attn.q_norm.weight', 'self_attn.q_layernorm.weight'], precision, f'layer_{i}_attn_q_norm.weights', False),
            (['self_attn.k_norm.weight', 'self_attn.k_layernorm.weight'], precision, f'layer_{i}_attn_k_norm.weights', False),
            (['mlp.gate_proj.weight', 'mlp.c_fc.weight'], precision, f'layer_{i}_ffn_gate.weights', False),
            (['mlp.up_proj.weight'], precision, f'layer_{i}_ffn_up.weights', False),
            (['mlp.down_proj.weight', 'mlp.c_proj.weight'], precision, f'layer_{i}_ffn_down.weights', False),
            (['post_attention_layernorm.weight', 'ln_2.weight'], precision, f'layer_{i}_post_attn_norm.weights', False),
        ]
        
        for name_patterns, tensor_precision, output_name, should_transpose in weight_patterns:
            found = False
            for pattern in name_patterns:
                full_name = layer_prefix + pattern
                if full_name in state_dict:
                    tensor = state_dict[full_name]
                    save_tensor_with_header(tensor, output_dir / output_name, tensor_precision, transpose=should_transpose, stats_tracker=quantization_stats, args=args)
                    found = True
                    break
            
            if not found and 'c_attn.weight' in name_patterns[0]:
                attn_name = layer_prefix + 'attn.c_attn.weight'
                if attn_name in state_dict:
                    combined_weight = state_dict[attn_name]
                    hidden_size = combined_weight.shape[0]
                    q_weight = combined_weight[:, :hidden_size]
                    k_weight = combined_weight[:, hidden_size:2*hidden_size]
                    v_weight = combined_weight[:, 2*hidden_size:]
                    
                    save_tensor_with_header(q_weight, output_dir / f'layer_{i}_attn_q.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args)
                    save_tensor_with_header(k_weight, output_dir / f'layer_{i}_attn_k.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args)
                    save_tensor_with_header(v_weight, output_dir / f'layer_{i}_attn_v.weights', precision, transpose=False, stats_tracker=quantization_stats, args=args)
                    found = True
            
    
    if quantization_stats['quantized_tensors'] > 0:
        mse_values = np.array(quantization_stats['mse_values'])
        snr_values = np.array(quantization_stats['snr_values'])
        cos_sim_values = np.array(quantization_stats['cos_sim_values'])
        
        print("\nQuantization Summary:")
        print(f"MSE - Mean: {np.mean(mse_values):.2e}, Max: {np.max(mse_values):.2e}, Median: {np.median(mse_values):.2e}, Min: {np.min(mse_values):.2e}")
        print(f"SNR - Mean: {np.mean(snr_values):.1f}dB, Max: {np.max(snr_values):.1f}dB, Median: {np.median(snr_values):.1f}dB, Min: {np.min(snr_values):.1f}dB")
        print(f"CosSim - Mean: {np.mean(cos_sim_values):.6f}, Max: {np.mean(cos_sim_values):.6f}, Median: {np.median(cos_sim_values):.6f}, Min: {np.min(cos_sim_values):.6f}")
        fp16_tensors = quantization_stats['total_tensors'] - quantization_stats['quantized_tensors']
        low_snr_fallbacks = quantization_stats.get('low_snr_fallbacks', 0)
        snr_threshold = args.snr_threshold if args else 30.0
        print(f"Processed {quantization_stats['quantized_tensors']} INT8 tensors, {fp16_tensors} FP16 tensors ({low_snr_fallbacks} SNR<{snr_threshold}dB fallbacks)")
    
    return model_config

def convert_hf_tokenizer(tokenizer, output_dir):
    is_sentencepiece = False
    tokenizer_model_path = None

    if hasattr(tokenizer, 'vocab_file'):
        vocab_file = tokenizer.vocab_file
        if vocab_file and vocab_file.endswith('.model'):
            is_sentencepiece = True
            tokenizer_model_path = vocab_file

    if not is_sentencepiece and hasattr(tokenizer, 'sp_model'):
        is_sentencepiece = True
        try:
            from huggingface_hub import hf_hub_download
            tokenizer_model_path = hf_hub_download(
                repo_id=tokenizer.name_or_path,
                filename="tokenizer.model"
            )
        except:
            pass

    if is_sentencepiece and tokenizer_model_path:
        import shutil
        dest_path = output_dir / "tokenizer.model"
        try:
            shutil.copy2(tokenizer_model_path, dest_path)
            print(f"  Copied SentencePiece model to {dest_path.name}")
        except Exception as e:
            print(f"  Warning: Could not copy tokenizer.model: {e}")

    vocab = tokenizer.get_vocab()

    id_to_token = [""] * len(vocab)
    for token, token_id in vocab.items():
        if token_id < len(id_to_token):
            id_to_token[token_id] = token

    vocab_output = output_dir / "vocab.txt"

    if is_sentencepiece:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token_id, token in enumerate(id_to_token):
                if token:
                    f.write(f"{token_id}\t{token}\n")
        print(f"  Saved SentencePiece vocabulary (ID\\ttoken format)")
    else:
        with open(vocab_output, 'w', encoding='utf-8') as f:
            for token in id_to_token:
                f.write(token + '\n')
        print(f"  Saved BPE vocabulary (line-by-line format)")
    
    
    merges_output = output_dir / "merges.txt"
    try:
        try:
            from huggingface_hub import hf_hub_download
            merges_file = hf_hub_download(repo_id=tokenizer.name_or_path, filename="merges.txt")
            import shutil
            shutil.copy2(merges_file, merges_output)
        except Exception:
            if hasattr(tokenizer, 'backend_tokenizer') and tokenizer.backend_tokenizer:
                backend = tokenizer.backend_tokenizer
                vocab = backend.get_vocab()
                merges = []
                
                if hasattr(backend, 'model'):
                    model = backend.model
                    if hasattr(model, 'merges'):
                        merges = model.merges
                
                if merges:
                    with open(merges_output, 'w', encoding='utf-8') as f:
                        f.write("#version: 0.2\n")
                        for merge in merges:
                            f.write(f"{merge}\n")
                else:
                    with open(merges_output, 'w', encoding='utf-8') as f:
                        f.write("#version: 0.2\n")
            else:
                with open(merges_output, 'w', encoding='utf-8') as f:
                    f.write("#version: 0.2\n")
    except Exception:
        with open(merges_output, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
    
    
    special_tokens = {}
    special_token_ids = {}
    
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        special_token_ids['eos_token_id'] = tokenizer.eos_token_id
        special_tokens[tokenizer.eos_token_id] = tokenizer.eos_token or "<|endoftext|>"
    
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        special_token_ids['pad_token_id'] = tokenizer.pad_token_id
        special_tokens[tokenizer.pad_token_id] = tokenizer.pad_token or "<|endoftext|>"
    
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        special_token_ids['bos_token_id'] = tokenizer.bos_token_id
        special_tokens[tokenizer.bos_token_id] = tokenizer.bos_token or "<|startoftext|>"
    
    if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
        special_token_ids['unk_token_id'] = tokenizer.unk_token_id
        special_tokens[tokenizer.unk_token_id] = tokenizer.unk_token or "<|unknown|>"
    
    additional_special_tokens = []
    if hasattr(tokenizer, 'additional_special_tokens'):
        for token in tokenizer.additional_special_tokens or []:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                special_tokens[token_id] = token
                additional_special_tokens.append({"token": token, "id": token_id})
    
    chat_template_data = {}
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        chat_template_output = output_dir / "chat_template.jinja2"
        with open(chat_template_output, 'w', encoding='utf-8') as f:
            f.write(tokenizer.chat_template)
        chat_template_data["chat_template"] = tokenizer.chat_template
    
    tokenizer_full_config = {}
    added_tokens_decoder = {}
    tool_tokens = {}
    
    try:
        config_path = None
        if hasattr(tokenizer, 'name_or_path'):
            from huggingface_hub import hf_hub_download
            try:
                config_path = hf_hub_download(repo_id=tokenizer.name_or_path, filename="tokenizer_config.json")
                with open(config_path, 'r') as f:
                    tokenizer_full_config = json.load(f)
                    
                    if 'chat_template' in tokenizer_full_config and not chat_template_data:
                        chat_template_output = output_dir / "chat_template.jinja2"
                        with open(chat_template_output, 'w', encoding='utf-8') as f:
                            f.write(tokenizer_full_config['chat_template'])
                        chat_template_data["chat_template"] = tokenizer_full_config['chat_template']
                    
                    if 'added_tokens_decoder' in tokenizer_full_config:
                        added_tokens_decoder = tokenizer_full_config['added_tokens_decoder']
                        
                        print("  Extracting special tokens from tokenizer_config.json...")
                        for token_id_str, token_info in added_tokens_decoder.items():
                            content = token_info.get('content', '')
                            token_id = int(token_id_str)
                            
                            tool_related = ['<tool_call>', '</tool_call>', 
                                          '<tool_response>', '</tool_response>',
                                          '<tools>', '</tools>',
                                          '<think>', '</think>']
                            
                            if any(x == content for x in tool_related):
                                tool_tokens[token_id] = token_info
                                print(f"    Found tool token: {content} (ID: {token_id})")
                                special_tokens[token_id] = content
                                
            except Exception as e:
                print(f"  Note: Could not load full tokenizer config: {e}")
                pass
    except Exception:
        pass
    
    
    special_tokens_output = output_dir / "special_tokens.json"
    with open(special_tokens_output, 'w', encoding='utf-8') as f:
        json.dump({
            **special_token_ids,
            "vocab_size": len(vocab),
            "model_max_length": getattr(tokenizer, 'model_max_length', 131072),
            "special_tokens": special_tokens,
            "additional_special_tokens": additional_special_tokens,
            **chat_template_data
        }, f, indent=2, ensure_ascii=False)
    
    
    tokenizer_config_output = output_dir / "tokenizer_config.txt"
    with open(tokenizer_config_output, 'w') as f:
        f.write(f"vocab_size={len(vocab)}\n")
        for key, value in special_token_ids.items():
            f.write(f"{key}={value}\n")
        f.write(f"model_max_length={getattr(tokenizer, 'model_max_length', 131072)}\n")

        if is_sentencepiece:
            f.write("tokenizer_type=sentencepiece\n")
        else:
            f.write("tokenizer_type=bpe\n")

        if chat_template_data:
            f.write("has_chat_template=true\n")
        else:
            f.write("has_chat_template=false\n")
        if len(tool_tokens) > 0:
            f.write(f"has_tool_support=true\n")
            f.write(f"tool_token_count={len(tool_tokens)}\n")
    

def convert_hf_to_cactus(model_name, output_dir, precision='INT8', cache_dir=None, args=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {model_name} to {precision}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    config = convert_hf_model_weights(model, output_dir, precision, args)
    
    if precision == 'INT8':
        config['precision'] = "FP16"
    else:
        config['precision'] = precision
    
    config_path = output_dir / "config.txt"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            if isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            f.write(f"{key}={value_str}\n")
    
    convert_hf_tokenizer(tokenizer, output_dir)
    print(f"\nConversion complete: {output_dir}")
    
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_parser():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace models to Cactus format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('model_name', help='HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")')
    parser.add_argument('output_dir', help='Directory to write converted files')
    parser.add_argument('--precision', choices=['INT8', 'FP16', 'FP32'], default='INT8',
                       help='Quantization precision')
    parser.add_argument('--cache-dir', help='Cache directory for HuggingFace models')
    
    quant_group = parser.add_argument_group('Quantization Parameters')
    quant_group.add_argument('--snr-threshold', type=float, default=20.0,
                            help='Minimum SNR (dB) for INT8 quantization, fallback to FP32 below this')
    quant_group.add_argument('--saturation-threshold', type=float, default=0.01,
                            help='Saturation threshold for outlier clipping (0.0-1.0)')
    quant_group.add_argument('--outlier-percentile', type=float, default=0.01,
                            help='Percentile for outlier detection (0.0-50.0)')
    quant_group.add_argument('--sigma-multiplier', type=float, default=3.5,
                            help='Standard deviation multiplier for range clipping')
    quant_group.add_argument('--range-threshold', type=float, default=0.5,
                            help='Minimum range preservation ratio (0.0-1.0)')
    quant_group.add_argument('--saturation-warning-threshold', type=float, default=0.1,
                            help='Saturation percentage threshold for warnings')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    if args.precision not in ['INT8', 'FP16', 'FP32']:
        print(f"Error: Invalid precision '{args.precision}'. Must be INT8, FP16, or FP32")
        sys.exit(1)
    
    convert_hf_to_cactus(args.model_name, args.output_dir, args.precision, args.cache_dir, args)