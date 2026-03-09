# LLMLingua-2 (DJL Implementation)

A Kotlin implementation of [LLMLingua-2](https://github.com/microsoft/LLMLingua) using [Deep Java Library (DJL)](https://djl.ai/) with ONNX Runtime for efficient prompt compression.

## Overview

LLMLingua-2 is a prompt compression technique proposed by Microsoft that uses a BERT-based model to identify and remove less important tokens from prompts while preserving semantic meaning. This implementation provides a JVM-based solution using DJL for inference.

## Requirements

- Java 21 or higher
- Gradle 8.x

## Setup

### 1. Prepare the ONNX Model

Export the published LLMLingua-2 checkpoint to ONNX and save the tokenizer files:

Note: The following steps require Python and `pip` to be installed.

```bash
# Create model directory
mkdir -p ~/llmlingua2_onnx

# Export the Microsoft checkpoint to ONNX
pip install "optimum[exporters]"
optimum-cli export onnx \
  --model microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
  --task token-classification \
  ~/llmlingua2_onnx
```

The model directory should contain:
- `model.onnx` - The ONNX model file
- `tokenizer.json` - The HuggingFace tokenizer
- `config.json` and tokenizer metadata generated during export

### 2. Build the Project

```bash
./gradlew build
```

## Usage

### Command Line

```bash
# Basic usage
./gradlew run --args="--text 'Your prompt text here' --rate 0.5"

# With custom model path
./gradlew run --args="--model-path /path/to/model --text 'Your prompt' --rate 0.5"

# Read from stdin
echo "Your long prompt text..." | ./gradlew run --args="--rate 0.5"
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path <path>` | Path to the LLMLingua-2 ONNX directory | `~/llmlingua2_onnx` |
| `--rate <0.0-1.0>` | Fraction of tokens to remove | `0.5` |
| `--text <prompt>` | Prompt to compress (stdin used if omitted) | - |
| `--force-token <token>` | Token to always preserve (repeatable) | `\n`, `?`, `!`, `.`, `,` |
| `--no-force-reserve-digits` | Allow numeric tokens to be removed | - |
| `--help` | Show help message | - |

### Programmatic Usage

```kotlin
import org.example.Llmlingua2Compressor
import java.nio.file.Paths

fun main() {
    val modelPath = Paths.get(System.getProperty("user.home"), "llmlingua2_onnx")
    
    Llmlingua2Compressor(modelPath).use { compressor ->
        val result = compressor.compress(
            text = "Your prompt text here...",
            reduceRate = 0.5,
            forceTokens = setOf("\n", "?", "!", ".", ","),
            forceReserveDigits = true
        )
        
        println(result.compressedText)
        println("Compression ratio: ${result.compressedTokenCount}/${result.originalTokenCount}")
    }
}
```

## How It Works

1. **Tokenization**: The input text is tokenized using the HuggingFace tokenizer
2. **Chunking**: Long texts are split into chunks respecting sentence boundaries
3. **Scoring**: Each token receives a "keep probability" from the BERT-based model
4. **Word Aggregation**: WordPiece tokens are aggregated into words, with scores based on maximum token probability
5. **Thresholding**: A dynamic threshold is computed based on the target compression rate
6. **Decoding**: Remaining tokens are decoded back to text

## Dependencies

- [DJL API](https://djl.ai/) (0.36.0) - Deep Java Library
- [ONNX Runtime Engine](https://djl.ai/docs/engines.html#onnx-runtime) - For model inference
- [HuggingFace Tokenizers](https://djl.ai/docs/tokenizers.html) - For text tokenization

## References

- [LLMLingua-2 Paper](https://arxiv.org/abs/2403.12968) - Pan et al., 2024
- [Microsoft LLMLingua](https://github.com/microsoft/LLMLingua) - Original Python implementation
- [DJL Documentation](https://djl.ai/docs/)

## License

The pretrained LLMLingua-2 model and related artifacts are subject to the license terms published with the original Microsoft release.

This repository does not currently include a separate `LICENSE` file for the Kotlin implementation code. If you intend to reuse or distribute this implementation, add an explicit repository license (for example, MIT or Apache-2.0) that is distinct from the model usage terms.
