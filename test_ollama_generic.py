#!/usr/bin/env python3
"""
Generic Ollama Model Testing Script for Crawl4AI
This script can test any Ollama model with configurable parameters via command-line arguments.
"""

import asyncio
import os
import json
import httpx
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig


class GeneralContent(BaseModel):
    """Generic schema for general content extraction"""
    title: str = Field(..., description="Main title or heading of the content")
    key_points: List[str] = Field(..., description="Main points or topics covered")
    summary: str = Field(..., description="Brief summary of the content")
    entities: List[str] = Field(..., description="Important entities, names, or concepts mentioned")


class CodeAnalysis(BaseModel):
    """Schema for analyzing code-related content"""
    programming_languages: List[str] = Field(..., description="Programming languages mentioned or used")
    frameworks_libraries: List[str] = Field(..., description="Frameworks or libraries mentioned")
    concepts: List[str] = Field(..., description="Programming concepts or paradigms discussed")
    difficulty_level: str = Field(..., description="Beginner, Intermediate, or Advanced")
    summary: str = Field(..., description="Brief summary of the technical content")


class TechnicalDocumentation(BaseModel):
    """Schema for extracting technical documentation"""
    title: str = Field(..., description="Title of the documentation")
    sections: List[str] = Field(..., description="Main sections or topics covered")
    code_examples: List[str] = Field(..., description="Code examples found in the content")
    installation_steps: List[str] = Field(..., description="Installation or setup steps if any")
    prerequisites: List[str] = Field(..., description="Prerequisites or requirements mentioned")


class NewsContent(BaseModel):
    """Schema for news content extraction"""
    headlines: List[str] = Field(..., description="Main headlines from the content")
    topics: List[str] = Field(..., description="Main topics or categories covered")
    key_facts: List[str] = Field(..., description="Important facts or statistics mentioned")
    summary: str = Field(..., description="Summary of the news content")


# Test configurations for different model types
MODEL_PRESETS = {
    "coding": {
        "schema": CodeAnalysis,
        "instruction": "Analyze this content for programming and technical information. Focus on languages, frameworks, concepts, and difficulty level.",
        "test_urls": [
            "https://docs.python.org/3/tutorial/",
            "https://reactjs.org/docs/getting-started.html",
            "https://docs.docker.com/get-started/"
        ]
    },
    "documentation": {
        "schema": TechnicalDocumentation,
        "instruction": "Extract structured information from this technical documentation. Focus on sections, code examples, and setup instructions.",
        "test_urls": [
            "https://docs.docker.com/get-started/",
            "https://docs.github.com/en/get-started",
            "https://kubernetes.io/docs/concepts/"
        ]
    },
    "news": {
        "schema": NewsContent,
        "instruction": "Extract news information including headlines, topics, key facts, and provide a summary.",
        "test_urls": [
            "https://www.reuters.com/technology/",
            "https://techcrunch.com/",
            "https://www.bbc.com/news/technology"
        ]
    },
    "general": {
        "schema": GeneralContent,
        "instruction": "Extract the main information from this content including title, key points, summary, and important entities.",
        "test_urls": [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://www.example.com",
            "https://github.com/microsoft/TypeScript"
        ]
    }
}


async def check_ollama_connection(ollama_url: str, model_name: str = None) -> tuple[bool, List[str]]:
    """
    Check if Ollama server is accessible and list available models
    
    Args:
        ollama_url: Base URL of the Ollama server
        model_name: Optional specific model to check for
    
    Returns:
        Tuple of (connection_success, available_models)
    """
    print(f"🔍 Checking Ollama connection at {ollama_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                print("✅ Ollama server is accessible")
                print(f"Available models ({len(model_names)}): {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}")
                
                if model_name:
                    # Check for exact match or partial match
                    exact_matches = [name for name in model_names if name == model_name]
                    partial_matches = [name for name in model_names if model_name in name or name in model_name]
                    
                    if exact_matches:
                        print(f"✅ Found exact model match: {exact_matches[0]}")
                        return True, model_names
                    elif partial_matches:
                        print(f"✅ Found similar model(s): {partial_matches}")
                        return True, model_names
                    else:
                        print(f"⚠️ Model '{model_name}' not found")
                        print("Available models:")
                        for name in model_names:
                            print(f"  - {name}")
                        return False, model_names
                
                return True, model_names
            else:
                print(f"❌ Ollama server responded with status {response.status_code}")
                return False, []
                
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {str(e)}")
        return False, []


async def test_ollama_model(model_name: str, ollama_url: str, url: str, 
                           schema: BaseModel, instruction: str,
                           extraction_type: str = "schema",
                           temperature: float = 0.7, max_tokens: int = 2000,
                           timeout: int = 60, verbose: bool = False) -> Dict[str, Any]:
    """
    Test an Ollama model with given parameters
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"URL: {url}")
        print(f"Schema: {schema.__name__}")
        print(f"Extraction: {extraction_type}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=verbose
        )
        
        # Configure LLM for Ollama
        llm_config = LLMConfig(
            provider=f"ollama/{model_name}" if not model_name.startswith("ollama/") else model_name,
            api_token="no-token-needed",
            base_url=ollama_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Configure extraction strategy
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            extraction_type=extraction_type,
            instruction=instruction,
            schema=schema.model_json_schema() if schema else None,
            extra_args={
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": max_tokens
            }
        )
        
        # Configure crawler
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            word_count_threshold=50,
            page_timeout=timeout * 1000,  # Convert to milliseconds
            verbose=verbose
        )
        
        # Run crawl
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.success:
                if verbose:
                    print(f"✅ Success! ({execution_time:.2f}s)")
                    print(f"Markdown length: {len(result.markdown.raw_markdown)}")
                
                # Parse extracted content
                try:
                    extracted_data = json.loads(result.extracted_content) if result.extracted_content else {}
                    if verbose and extracted_data:
                        print("📊 Extracted data preview:")
                        for key, value in list(extracted_data.items())[:3]:
                            if isinstance(value, list):
                                print(f"  {key}: {len(value)} items")
                            elif isinstance(value, str):
                                preview = value[:100] + "..." if len(value) > 100 else value
                                print(f"  {key}: {preview}")
                except json.JSONDecodeError:
                    extracted_data = {"raw_content": result.extracted_content}
                
                return {
                    "model": model_name,
                    "url": url,
                    "success": True,
                    "execution_time": execution_time,
                    "markdown_length": len(result.markdown.raw_markdown),
                    "extracted_data": extracted_data,
                    "schema_used": schema.__name__,
                    "extraction_type": extraction_type
                }
            else:
                if verbose:
                    print(f"❌ Failed: {result.error_message}")
                return {
                    "model": model_name,
                    "url": url,
                    "success": False,
                    "execution_time": execution_time,
                    "error": result.error_message,
                    "schema_used": schema.__name__,
                    "extraction_type": extraction_type
                }
                
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        if verbose:
            print(f"❌ Exception: {str(e)}")
        return {
            "model": model_name,
            "url": url,
            "success": False,
            "execution_time": execution_time,
            "error": str(e),
            "schema_used": schema.__name__ if schema else "None",
            "extraction_type": extraction_type
        }


def load_urls_from_file(file_path: str) -> List[str]:
    """Load URLs from a text file (one URL per line)"""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    except Exception as e:
        print(f"❌ Error loading URLs from {file_path}: {e}")
        return []


def save_results(results: List[Dict], output_file: str, format_type: str = "json"):
    """Save results in various formats"""
    try:
        if format_type == "json":
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format_type == "csv":
            import csv
            if results:
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        elif format_type == "markdown":
            with open(output_file, 'w') as f:
                f.write("# Ollama Model Test Results\n\n")
                for result in results:
                    f.write(f"## {result.get('model', 'Unknown')} - {result.get('url', 'Unknown URL')}\n")
                    f.write(f"- **Success**: {result.get('success', False)}\n")
                    f.write(f"- **Execution Time**: {result.get('execution_time', 0):.2f}s\n")
                    if result.get('success'):
                        f.write(f"- **Markdown Length**: {result.get('markdown_length', 0)} chars\n")
                        f.write(f"- **Schema**: {result.get('schema_used', 'Unknown')}\n")
                    else:
                        f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
                    f.write("\n")
        
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generic Ollama Model Testing Script for Crawl4AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with any model
  python test_ollama_generic.py --model llama3 --url https://example.com

  # Test coding model on technical content
  python test_ollama_generic.py --model devstral-small-2505 --test-type coding

  # Batch test multiple URLs
  python test_ollama_generic.py --model codellama --urls-file tech_sites.txt

  # Custom configuration
  python test_ollama_generic.py --model mistral --temperature 0.3 --max-tokens 1000

  # List available models
  python test_ollama_generic.py --list-models
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        help="Ollama model name (e.g., llama3, devstral-small-2505, codellama)"
    )
    
    # URL arguments (mutually exclusive)
    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument(
        "--url", "-u",
        help="Single URL to test"
    )
    url_group.add_argument(
        "--urls-file",
        help="File containing list of URLs to test (one per line)"
    )
    
    # Test configuration
    parser.add_argument(
        "--test-type", "-t",
        choices=["general", "coding", "documentation", "news", "all"],
        default="general",
        help="Type of test to run (default: general)"
    )
    
    parser.add_argument(
        "--extraction-type", "-e",
        choices=["instruction", "schema", "block"],
        default="schema",
        help="Extraction type (default: schema)"
    )
    
    # Ollama configuration
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    # LLM parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (0.0-1.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens for response (default: 2000)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Utility options
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models on Ollama server and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


async def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Generic Ollama Model Testing for Crawl4AI")
    print("=" * 50)
    
    # Check if we should just list models
    if args.list_models:
        success, models = await check_ollama_connection(args.ollama_url)
        if success:
            print(f"\n📋 Available models on {args.ollama_url}:")
            for i, model in enumerate(models, 1):
                print(f"  {i:2d}. {model}")
        return
    
    # Validate required arguments
    if not args.model:
        print("❌ Error: --model is required (use --list-models to see available models)")
        return
    
    # Check Ollama connection
    connection_ok, available_models = await check_ollama_connection(args.ollama_url, args.model)
    if not connection_ok:
        print(f"\n❌ Cannot connect to Ollama or model '{args.model}' not found")
        print("Available models:")
        for model in available_models[:10]:  # Show first 10
            print(f"  - {model}")
        return
    
    # Determine URLs to test
    urls_to_test = []
    
    if args.urls_file:
        urls_to_test = load_urls_from_file(args.urls_file)
        if not urls_to_test:
            print("❌ No valid URLs found in file")
            return
    elif args.url:
        urls_to_test = [args.url]
    else:
        # Use preset URLs based on test type
        if args.test_type == "all":
            for preset in MODEL_PRESETS.values():
                urls_to_test.extend(preset["test_urls"][:1])  # One URL from each type
        else:
            urls_to_test = MODEL_PRESETS[args.test_type]["test_urls"][:1]  # First URL from type
    
    print(f"\n🎯 Testing Model: {args.model}")
    print(f"📋 Test Type: {args.test_type}")
    print(f"🌐 URLs to test: {len(urls_to_test)}")
    
    # Run tests
    all_results = []
    
    if args.test_type == "all":
        # Test with all schema types
        test_types = ["general", "coding", "documentation", "news"]
        for test_type in test_types:
            preset = MODEL_PRESETS[test_type]
            url = urls_to_test[0] if urls_to_test else preset["test_urls"][0]
            
            if args.verbose:
                print(f"\n🔄 Running {test_type} test...")
            
            result = await test_ollama_model(
                model_name=args.model,
                ollama_url=args.ollama_url,
                url=url,
                schema=preset["schema"],
                instruction=preset["instruction"],
                extraction_type=args.extraction_type,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                verbose=args.verbose
            )
            result["test_type"] = test_type
            all_results.append(result)
    else:
        # Test with specific schema type
        preset = MODEL_PRESETS[args.test_type]
        
        for i, url in enumerate(urls_to_test, 1):
            if args.verbose or len(urls_to_test) > 1:
                print(f"\n🔄 Testing URL {i}/{len(urls_to_test)}: {url}")
            
            result = await test_ollama_model(
                model_name=args.model,
                ollama_url=args.ollama_url,
                url=url,
                schema=preset["schema"],
                instruction=preset["instruction"],
                extraction_type=args.extraction_type,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                verbose=args.verbose
            )
            result["test_type"] = args.test_type
            all_results.append(result)
    
    # Display summary
    print("\n" + "="*50)
    print("📈 TEST SUMMARY")
    print("="*50)
    
    successful_tests = [r for r in all_results if r.get('success', False)]
    failed_tests = [r for r in all_results if not r.get('success', True)]
    
    print(f"✅ Successful tests: {len(successful_tests)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_time = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average execution time: {avg_time:.2f}s")
        
        fastest = min(successful_tests, key=lambda x: x['execution_time'])
        slowest = max(successful_tests, key=lambda x: x['execution_time'])
        print(f"🏆 Fastest: {fastest['execution_time']:.2f}s ({fastest['url']})")
        print(f"🐌 Slowest: {slowest['execution_time']:.2f}s ({slowest['url']})")
    
    if failed_tests:
        print("\n⚠️ Failed tests:")
        for result in failed_tests:
            print(f"  - {result['url']}: {result.get('error', 'Unknown error')}")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = int(time.time())
        safe_model_name = args.model.replace("/", "_").replace(":", "_")
        output_file = f"ollama_test_{safe_model_name}_{timestamp}.{args.format}"
    
    save_results(all_results, output_file, args.format)
    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())