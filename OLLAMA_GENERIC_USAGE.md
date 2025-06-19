# Generic Ollama Testing Script Usage Guide

The `test_ollama_generic.py` script allows you to test any Ollama model with Crawl4AI using command-line arguments.

## Quick Start

### 1. List Available Models
```bash
python test_ollama_generic.py --list-models --ollama-url http://your-server:11434
```

### 2. Basic Test
```bash
# Test any model with default settings
python test_ollama_generic.py --model llama3 --url https://example.com

# Test with your remote Ollama server
python test_ollama_generic.py --model devstral-small-2505 --ollama-url http://your-server:11434
```

### 3. Test Type Presets
```bash
# Test coding capabilities (great for Devstral)
python test_ollama_generic.py --model devstral-small-2505 --test-type coding

# Test documentation extraction
python test_ollama_generic.py --model llama3 --test-type documentation

# Test news extraction
python test_ollama_generic.py --model mistral --test-type news

# Test all types
python test_ollama_generic.py --model codellama --test-type all
```

### 4. Batch Testing
```bash
# Test multiple URLs from a file
python test_ollama_generic.py --model llama3 --urls-file test_urls.txt

# Custom URL list
echo -e "https://docs.python.org/3/tutorial/\nhttps://reactjs.org/docs/" > my_urls.txt
python test_ollama_generic.py --model devstral-small-2505 --urls-file my_urls.txt
```

## Command-Line Options

### Required
- `--model, -m` - Ollama model name (e.g., `llama3`, `devstral-small-2505`, `codellama`)

### URLs (choose one)
- `--url, -u` - Single URL to test
- `--urls-file` - File with URLs (one per line)

### Test Configuration
- `--test-type, -t` - Test type: `general`, `coding`, `documentation`, `news`, `all` (default: `general`)
- `--extraction-type, -e` - Extraction method: `instruction`, `schema`, `block` (default: `schema`)

### Server Configuration
- `--ollama-url` - Ollama server URL (default: `http://localhost:11434`)

### LLM Parameters
- `--temperature` - Creativity level 0.0-1.0 (default: 0.7)
- `--max-tokens` - Maximum response length (default: 2000)
- `--timeout` - Request timeout in seconds (default: 60)

### Output Options
- `--output, -o` - Output file name (auto-generated if not specified)
- `--format` - Output format: `json`, `csv`, `markdown` (default: `json`)

### Utility
- `--list-models` - Show available models and exit
- `--verbose, -v` - Detailed output during testing

## Test Type Schemas

### General Content
Extracts: title, key_points, summary, entities

### Coding Analysis
Extracts: programming_languages, frameworks_libraries, concepts, difficulty_level, summary

### Technical Documentation
Extracts: title, sections, code_examples, installation_steps, prerequisites

### News Content
Extracts: headlines, topics, key_facts, summary

## Examples

### Testing Devstral for Code Analysis
```bash
# Test Devstral's coding capabilities
python test_ollama_generic.py \
    --model devstral-small-2505 \
    --test-type coding \
    --ollama-url http://your-server:11434 \
    --verbose

# Compare multiple coding models
python test_ollama_generic.py --model devstral-small-2505 --test-type coding --output devstral_coding.json
python test_ollama_generic.py --model codellama --test-type coding --output codellama_coding.json
```

### Performance Testing
```bash
# Test with different temperature settings
python test_ollama_generic.py --model llama3 --temperature 0.1 --output conservative.json
python test_ollama_generic.py --model llama3 --temperature 0.9 --output creative.json

# Quick vs detailed responses
python test_ollama_generic.py --model mistral --max-tokens 500 --output quick.json
python test_ollama_generic.py --model mistral --max-tokens 2000 --output detailed.json
```

### Batch Content Analysis
```bash
# Test multiple technical sites
python test_ollama_generic.py \
    --model devstral-small-2505 \
    --urls-file test_urls.txt \
    --test-type coding \
    --format markdown \
    --output tech_analysis.md
```

### Custom URL Lists
```bash
# Create custom test set
cat > ai_sites.txt << EOF
https://openai.com/blog/
https://www.anthropic.com/research
https://ai.google/research/
https://www.deepmind.com/research
EOF

# Test AI content extraction
python test_ollama_generic.py \
    --model llama3 \
    --urls-file ai_sites.txt \
    --test-type general \
    --verbose
```

## Output Files

Results are saved in the specified format:

### JSON (default)
```json
[
  {
    "model": "devstral-small-2505",
    "url": "https://docs.python.org/3/tutorial/",
    "success": true,
    "execution_time": 3.45,
    "extracted_data": {...},
    "schema_used": "CodeAnalysis",
    "test_type": "coding"
  }
]
```

### CSV
Tabular format with all fields, good for spreadsheet analysis.

### Markdown
Human-readable report format with test summaries.

## Integration with Existing Scripts

You can use this generic script alongside the specific ones:

```bash
# Compare Gemini vs Generic Ollama model
python test_gemini_integration.py  # Tests Gemini models
python test_ollama_generic.py --model llama3 --test-type all  # Tests Ollama model

# Test multiple Ollama models
for model in llama3 mistral codellama devstral-small-2505; do
    python test_ollama_generic.py --model $model --test-type coding --output ${model}_results.json
done
```

## Troubleshooting

### Model Not Found
```bash
# List available models first
python test_ollama_generic.py --list-models --ollama-url http://your-server:11434

# Pull model if needed (on your Ollama server)
ollama pull devstral-small-2505
```

### Connection Issues
```bash
# Test connection
curl http://your-server:11434/api/tags

# Check if server is accessible
python test_ollama_generic.py --list-models --ollama-url http://your-server:11434
```

### Performance Issues
```bash
# Reduce token limit for faster responses
python test_ollama_generic.py --model llama3 --max-tokens 500

# Increase timeout for slow models
python test_ollama_generic.py --model large-model --timeout 120
```

This generic script provides maximum flexibility for testing any Ollama model with Crawl4AI!