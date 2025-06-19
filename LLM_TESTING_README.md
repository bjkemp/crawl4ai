# LLM Testing with Crawl4AI: Gemini vs Devstral

This directory contains test scripts for comparing Google Gemini and Mistral's Devstral models with Crawl4AI.

## 📁 Files Overview

- `test_gemini_integration.py` - Comprehensive testing of Google Gemini models
- `test_devstral_integration.py` - Testing Devstral via Ollama on remote server
- `test_llm_comparison.py` - Side-by-side comparison of both models
- `setup_test_env.py` - Environment setup and validation script

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script to check your environment
python setup_test_env.py
```

### 2. Configure API Keys

Create a `.env` file or set environment variables:

```bash
# Google Gemini API Key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Remote Ollama Server URL
export OLLAMA_BASE_URL="http://your-remote-server:11434"
```

### 3. Install Devstral on Remote Server

On your remote server running Ollama:

```bash
# Pull the Devstral model
ollama pull devstral-small-2505

# Verify it's installed
ollama list
```

### 4. Run Tests

```bash
# Test individual models
python test_gemini_integration.py
python test_devstral_integration.py

# Compare both models
python test_llm_comparison.py
```

## 🧪 Test Descriptions

### Gemini Integration Tests (`test_gemini_integration.py`)

Tests various Gemini models with different extraction strategies:

- **Basic Instruction Extraction**: Tests `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-pro`
- **Schema-Based Extraction**: Structured data extraction with Pydantic models
- **Block Extraction**: Extracting information blocks from content
- **Error Handling**: Testing with invalid configurations

**Sample Output:**
```
✅ Success! Markdown length: 1234
Extracted content preview:
{
  "headlines": ["AI News Headlines..."],
  "topics": ["Artificial Intelligence", "Technology"],
  "summary": "Latest developments in AI..."
}
```

### Devstral Integration Tests (`test_devstral_integration.py`)

Tests Mistral's Devstral model via Ollama with focus on coding tasks:

- **Code Analysis**: Analyzing programming content (Devstral's specialty)
- **Technical Documentation**: Extracting from developer docs
- **General Web Content**: Testing on non-coding content for comparison
- **Model Variations**: Testing different Devstral model names/versions

**Sample Output:**
```
✅ Found Devstral model(s): ['devstral-small-2505:latest']
✅ Success! Markdown length: 2345
Extracted content preview:
{
  "programming_languages": ["Python", "JavaScript"],
  "frameworks_libraries": ["React", "Django"],
  "concepts": ["async/await", "REST APIs"],
  "difficulty_level": "Intermediate"
}
```

### Comparison Tests (`test_llm_comparison.py`)

Direct comparison between Gemini and Devstral on identical tasks:

- **News Extraction**: Testing both models on news websites
- **Technical Content Analysis**: Comparing coding/tech analysis capabilities
- **Performance Metrics**: Execution time, success rates, output quality

**Sample Output:**
```
📊 DETAILED COMPARISON ANALYSIS
🏆 Fastest: Devstral Small 2505 (2.34s)
🐌 Slowest: Gemini 2.0 Flash (3.45s)
⚡ Speed difference: 1.11s (47.4% faster)

🎯 Model Success Rates:
  Gemini 2.0 Flash: 2/2 (100.0%)
  Devstral Small 2505: 2/2 (100.0%)
```

## 🔧 Configuration Details

### Gemini Models Supported

- `gemini/gemini-2.0-flash` - Latest multimodal model
- `gemini/gemini-1.5-pro` - Advanced reasoning model  
- `gemini/gemini-pro` - Standard model

### Devstral Models

- `devstral-small-2505` - Coding-specialized model (24B parameters)
- Supports various naming conventions: `ollama/devstral-small-2505`, `devstral:latest`, etc.

### Extraction Types

1. **Instruction-based**: Free-form text instructions
2. **Schema-based**: Structured extraction with Pydantic models
3. **Block-based**: Extracting semantic blocks of information

## 📊 Expected Results

### Gemini Strengths
- General knowledge and reasoning
- Multimodal capabilities
- Consistent performance across domains
- Good at following complex instructions

### Devstral Strengths  
- Specialized for software engineering tasks
- Code analysis and understanding
- Technical documentation processing
- Potentially faster inference (local deployment)

## 🐛 Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not set"**
   - Get API key from: https://aistudio.google.com/app/apikey
   - Set with: `export GEMINI_API_KEY="your_key"`

2. **"Cannot reach Ollama server"**
   - Ensure Ollama is running on remote server
   - Check firewall/network connectivity
   - Verify URL format: `http://server:11434`

3. **"No Devstral models found"**
   - Install on remote server: `ollama pull devstral-small-2505`
   - Check available models: `ollama list`

4. **Import errors**
   - Install dependencies: `pip install crawl4ai pydantic httpx`
   - Verify installation: `python -c "import crawl4ai; print('OK')"`

### Debug Mode

Add `verbose=True` to browser/crawler configs for detailed logging:

```python
browser_config = BrowserConfig(headless=True, verbose=True)
crawler_config = CrawlerRunConfig(verbose=True, ...)
```

## 📈 Performance Tips

1. **Caching**: Use `CacheMode.ENABLED` for repeated testing on same URLs
2. **Parallel Testing**: Run different model tests in parallel
3. **Timeout Settings**: Adjust `page_timeout` for slow websites
4. **Content Filtering**: Use `word_count_threshold` to focus on substantial content

## 🔬 Extending the Tests

To add your own tests:

1. Create new Pydantic schemas for your extraction needs
2. Add test functions following the existing patterns
3. Include both models in your comparison
4. Add custom instructions tailored to your use case

Example:
```python
class CustomSchema(BaseModel):
    field1: str = Field(..., description="Description of field1")
    field2: List[str] = Field(..., description="List of items")

async def test_custom_extraction():
    # Your test implementation
    pass
```

## 📝 Output Files

Tests generate several output files:

- `gemini_test_results.json` - Detailed Gemini test results
- `devstral_test_results.json` - Detailed Devstral test results  
- `llm_comparison_results_[timestamp].json` - Comparison results
- `comparison_summary_[timestamp].json` - Executive summary

These files contain detailed metrics, extracted content, and performance data for analysis.