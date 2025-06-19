#!/usr/bin/env python3
"""
Test script for Devstral integration with Crawl4AI via Ollama
This script tests Mistral's Devstral model running on a remote Ollama server.
"""

import asyncio
import os
import json
import httpx
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig


class CodeAnalysis(BaseModel):
    """Schema for analyzing code-related content"""
    programming_languages: list = Field(..., description="Programming languages mentioned or used")
    frameworks_libraries: list = Field(..., description="Frameworks or libraries mentioned")
    concepts: list = Field(..., description="Programming concepts or paradigms discussed")
    difficulty_level: str = Field(..., description="Beginner, Intermediate, or Advanced")
    summary: str = Field(..., description="Brief summary of the technical content")


class TechnicalDocumentation(BaseModel):
    """Schema for extracting technical documentation"""
    title: str = Field(..., description="Title of the documentation")
    sections: list = Field(..., description="Main sections or topics covered")
    code_examples: list = Field(..., description="Code examples found in the content")
    installation_steps: list = Field(..., description="Installation or setup steps if any")
    prerequisites: list = Field(..., description="Prerequisites or requirements mentioned")


async def check_ollama_connection(ollama_url: str) -> bool:
    """
    Check if Ollama server is accessible and has Devstral model
    
    Args:
        ollama_url: Base URL of the Ollama server
    """
    print(f"🔍 Checking Ollama connection at {ollama_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check if server is up
            response = await client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                print("✅ Ollama server is accessible")
                print(f"Available models: {model_names}")
                
                # Check for Devstral models
                devstral_models = [name for name in model_names if "devstral" in name.lower()]
                if devstral_models:
                    print(f"✅ Found Devstral model(s): {devstral_models}")
                    return True
                else:
                    print("⚠️ No Devstral models found. Available models:")
                    for name in model_names:
                        print(f"  - {name}")
                    return False
            else:
                print(f"❌ Ollama server responded with status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {str(e)}")
        return False


async def test_devstral_model(model_name: str, ollama_url: str, url: str, 
                             extraction_type: str = "instruction", 
                             schema: BaseModel = None, instruction: str = None) -> Dict[str, Any]:
    """
    Test Devstral model with given parameters
    
    Args:
        model_name: Devstral model identifier (e.g., 'ollama/devstral-small-2505')
        ollama_url: Base URL of the Ollama server
        url: URL to crawl
        extraction_type: 'instruction', 'schema', or 'block'
        schema: Pydantic model for schema extraction
        instruction: Custom instruction for LLM
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with {extraction_type} extraction")
    print(f"URL: {url}")
    print(f"Ollama URL: {ollama_url}")
    print(f"{'='*60}")
    
    try:
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        # Configure LLM for Ollama
        llm_config = LLMConfig(
            provider=model_name,
            api_token="no-token-needed",  # Ollama doesn't need API token
            base_url=ollama_url,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Configure extraction strategy
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            extraction_type=extraction_type,
            instruction=instruction,
            schema=schema.model_json_schema() if schema else None,
            extra_args={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        )
        
        # Configure crawler
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy,
            word_count_threshold=50,
            verbose=False
        )
        
        # Run crawl
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            
            if result.success:
                print(f"✅ Success!")
                print(f"Markdown length: {len(result.markdown.raw_markdown)}")
                print(f"Extracted content preview:")
                
                # Parse and display extracted content
                try:
                    extracted = json.loads(result.extracted_content) if result.extracted_content else {}
                    print(json.dumps(extracted, indent=2)[:500] + "..." if len(str(extracted)) > 500 else json.dumps(extracted, indent=2))
                except json.JSONDecodeError:
                    print(result.extracted_content[:500] + "..." if len(result.extracted_content) > 500 else result.extracted_content)
                
                return {
                    "model": model_name,
                    "success": True,
                    "markdown_length": len(result.markdown.raw_markdown),
                    "extracted_content": result.extracted_content,
                    "token_usage": getattr(result, 'token_usage', None)
                }
            else:
                print(f"❌ Failed: {result.error_message}")
                return {
                    "model": model_name,
                    "success": False,
                    "error": result.error_message
                }
                
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }


async def test_code_analysis():
    """Test Devstral's coding capabilities with technical content"""
    print("\n💻 Testing Code Analysis (Devstral's Specialty)")
    
    coding_urls = [
        "https://docs.python.org/3/tutorial/",
        "https://github.com/microsoft/TypeScript",
        "https://reactjs.org/docs/getting-started.html"
    ]
    
    instruction = """
    Analyze this technical content for programming-related information. 
    Focus on identifying programming languages, frameworks, concepts, and the overall technical level.
    As a coding-specialized model, provide insights about the development practices and technologies mentioned.
    """
    
    results = []
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    for url in coding_urls[:1]:  # Test with first URL to start
        result = await test_devstral_model(
            model_name="ollama/devstral-small-2505",
            ollama_url=ollama_url,
            url=url,
            extraction_type="schema",
            schema=CodeAnalysis,
            instruction=instruction
        )
        results.append(result)
    
    return results


async def test_technical_documentation():
    """Test extraction from technical documentation"""
    print("\n📚 Testing Technical Documentation Extraction")
    
    tech_docs_url = "https://docs.docker.com/get-started/"
    instruction = """
    Extract structured information from this technical documentation.
    Focus on identifying main sections, code examples, installation steps, and prerequisites.
    Provide a comprehensive analysis suitable for developers.
    """
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    result = await test_devstral_model(
        model_name="ollama/devstral-small-2505",
        ollama_url=ollama_url,
        url=tech_docs_url,
        extraction_type="schema",
        schema=TechnicalDocumentation,
        instruction=instruction
    )
    
    return [result]


async def test_general_web_content():
    """Test with non-coding content to compare performance"""
    print("\n🌐 Testing General Web Content")
    
    general_url = "https://en.wikipedia.org/wiki/Machine_learning"
    instruction = """
    Extract key information from this content about machine learning.
    Identify main concepts, applications, and any technical details mentioned.
    """
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    result = await test_devstral_model(
        model_name="ollama/devstral-small-2505",
        ollama_url=ollama_url,
        url=general_url,
        extraction_type="instruction",
        instruction=instruction
    )
    
    return [result]


async def test_model_variations():
    """Test different Devstral model variants if available"""
    print("\n🔄 Testing Different Model Variants")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Possible Devstral model names to try
    possible_models = [
        "ollama/devstral-small-2505",
        "ollama/devstral",
        "ollama/devstral:latest",
        "devstral-small-2505",
        "devstral"
    ]
    
    test_url = "https://github.com/microsoft/TypeScript/blob/main/README.md"
    instruction = "Analyze this GitHub README for technical information, setup instructions, and development details."
    
    results = []
    for model in possible_models[:2]:  # Test first 2 variants
        result = await test_devstral_model(
            model_name=model,
            ollama_url=ollama_url,
            url=test_url,
            extraction_type="instruction",
            instruction=instruction
        )
        results.append(result)
        
        # If first model works, we can skip testing others
        if result.get('success', False):
            break
    
    return results


def check_environment():
    """Check environment setup for Ollama connection"""
    print("🔧 Checking Environment Setup")
    
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if not ollama_url:
        print("⚠️ OLLAMA_BASE_URL not set, using default: http://localhost:11434")
        ollama_url = "http://localhost:11434"
    else:
        print(f"✅ OLLAMA_BASE_URL is set: {ollama_url}")
    
    return ollama_url


async def main():
    """Main test function"""
    print("🚀 Starting Crawl4AI + Devstral (Ollama) Integration Tests")
    print("=" * 70)
    
    # Check environment
    ollama_url = check_environment()
    
    # Check Ollama connection
    if not await check_ollama_connection(ollama_url):
        print("\n❌ Cannot proceed without Ollama connection.")
        print("Make sure:")
        print("1. Ollama is running on your remote server")
        print("2. Devstral model is installed (ollama pull devstral-small-2505)")
        print("3. OLLAMA_BASE_URL points to your server (e.g., http://your-server:11434)")
        return
    
    # Store all results
    all_results = []
    
    try:
        # Run all test suites
        code_results = await test_code_analysis()
        all_results.extend(code_results)
        
        docs_results = await test_technical_documentation()
        all_results.extend(docs_results)
        
        general_results = await test_general_web_content()
        all_results.extend(general_results)
        
        variant_results = await test_model_variations()
        all_results.extend(variant_results)
        
        # Summary
        print("\n" + "="*70)
        print("📈 TEST SUMMARY")
        print("="*70)
        
        successful_tests = [r for r in all_results if r.get('success', False)]
        failed_tests = [r for r in all_results if not r.get('success', True)]
        
        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        
        if successful_tests:
            print("\n🎉 Successful Models:")
            for result in successful_tests:
                print(f"  - {result['model']}")
        
        if failed_tests:
            print("\n⚠️ Failed Models:")
            for result in failed_tests:
                print(f"  - {result['model']}: {result.get('error', 'Unknown error')}")
        
        # Save detailed results
        with open("devstral_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: devstral_test_results.json")
        
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())