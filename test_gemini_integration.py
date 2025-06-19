#!/usr/bin/env python3
"""
Test script for Google Gemini integration with Crawl4AI
This script tests various Gemini models with different extraction strategies.
"""

import asyncio
import os
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig


class ProductInfo(BaseModel):
    """Schema for extracting product information"""
    name: str = Field(..., description="Product name")
    price: str = Field(..., description="Product price")
    description: str = Field(..., description="Product description")
    availability: str = Field(..., description="Product availability status")


class NewsArticle(BaseModel):
    """Schema for extracting news article information"""
    headline: str = Field(..., description="Main headline of the article")
    author: str = Field(..., description="Author of the article")
    published_date: str = Field(..., description="Publication date")
    summary: str = Field(..., description="Brief summary of the article")


async def test_gemini_model(model_name: str, url: str, extraction_type: str = "instruction", 
                           schema: BaseModel = None, instruction: str = None) -> Dict[str, Any]:
    """
    Test a specific Gemini model with given parameters
    
    Args:
        model_name: Gemini model identifier (e.g., 'gemini/gemini-2.0-flash')
        url: URL to crawl
        extraction_type: 'instruction', 'schema', or 'block'
        schema: Pydantic model for schema extraction
        instruction: Custom instruction for LLM
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with {extraction_type} extraction")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )
        
        # Configure LLM
        llm_config = LLMConfig(
            provider=model_name,
            api_token=os.getenv("GEMINI_API_KEY"),
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


async def test_basic_instruction_extraction():
    """Test basic instruction-based extraction"""
    print("\n🔍 Testing Basic Instruction Extraction")
    
    models_to_test = [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-pro"
    ]
    
    test_url = "https://www.example.com"
    instruction = "Extract the main content of this webpage including any important information, contact details, and key messages."
    
    results = []
    for model in models_to_test:
        result = await test_gemini_model(
            model_name=model,
            url=test_url,
            extraction_type="instruction",
            instruction=instruction
        )
        results.append(result)
    
    return results


async def test_schema_based_extraction():
    """Test schema-based extraction with structured data"""
    print("\n📊 Testing Schema-Based Extraction")
    
    # Test with news site
    news_url = "https://www.bbc.com/news"
    news_instruction = "Extract news articles from this page with their headlines, authors, dates, and summaries."
    
    result = await test_gemini_model(
        model_name="gemini/gemini-2.0-flash",
        url=news_url,
        extraction_type="schema",
        schema=NewsArticle,
        instruction=news_instruction
    )
    
    return [result]


async def test_block_extraction():
    """Test block-based extraction"""
    print("\n🧱 Testing Block Extraction")
    
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    instruction = "Extract key information blocks about artificial intelligence including definitions, history, applications, and current developments."
    
    result = await test_gemini_model(
        model_name="gemini/gemini-2.0-flash",
        url=test_url,
        extraction_type="block",
        instruction=instruction
    )
    
    return [result]


async def test_error_handling():
    """Test error handling with invalid configurations"""
    print("\n⚠️ Testing Error Handling")
    
    try:
        # Test with invalid API key
        invalid_llm_config = LLMConfig(
            provider="gemini/gemini-2.0-flash",
            api_token="invalid_key"
        )
        
        extraction_strategy = LLMExtractionStrategy(
            llm_config=invalid_llm_config,
            extraction_type="instruction",
            instruction="Test extraction"
        )
        
        crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=extraction_strategy
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url="https://www.example.com", config=crawler_config)
            
        print(f"Invalid API key test: {'Success' if not result.success else 'Unexpected success'}")
        
    except Exception as e:
        print(f"Caught expected error: {str(e)}")


def check_environment():
    """Check if required environment variables are set"""
    print("🔧 Checking Environment Setup")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your_api_key_here'")
        return False
    else:
        print("✅ GEMINI_API_KEY is set")
        print(f"Key preview: {gemini_key[:10]}...{gemini_key[-4:]}")
    
    return True


async def main():
    """Main test function"""
    print("🚀 Starting Crawl4AI + Google Gemini Integration Tests")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        return
    
    # Store all results
    all_results = []
    
    try:
        # Run all test suites
        basic_results = await test_basic_instruction_extraction()
        all_results.extend(basic_results)
        
        schema_results = await test_schema_based_extraction()
        all_results.extend(schema_results)
        
        block_results = await test_block_extraction()
        all_results.extend(block_results)
        
        await test_error_handling()
        
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
        with open("gemini_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: gemini_test_results.json")
        
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())