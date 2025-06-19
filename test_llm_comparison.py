#!/usr/bin/env python3
"""
Comparison test script for Google Gemini vs Devstral via Ollama with Crawl4AI
This script runs the same extraction tasks on both models and compares results.
"""

import asyncio
import os
import json
import time
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.types import LLMConfig


class NewsExtraction(BaseModel):
    """Schema for extracting news information"""
    headlines: List[str] = Field(..., description="Main headlines from the page")
    topics: List[str] = Field(..., description="Main topics or categories covered")
    summary: str = Field(..., description="Brief summary of the main news")


class TechContentAnalysis(BaseModel):
    """Schema for analyzing technical content"""
    technologies: List[str] = Field(..., description="Technologies or tools mentioned")
    concepts: List[str] = Field(..., description="Technical concepts discussed")
    complexity: str = Field(..., description="Content complexity: Beginner, Intermediate, or Advanced")
    summary: str = Field(..., description="Summary of the technical content")


async def run_extraction_test(model_config: Dict[str, Any], url: str, schema: BaseModel, 
                             instruction: str, test_name: str) -> Dict[str, Any]:
    """
    Run extraction test with a specific model configuration
    
    Args:
        model_config: Dictionary containing model configuration
        url: URL to crawl
        schema: Pydantic model for extraction
        instruction: Instruction for the LLM
        test_name: Name of the test for logging
    """
    print(f"\n🔄 Running {test_name} with {model_config['name']}")
    
    start_time = time.time()
    
    try:
        # Configure browser
        browser_config = BrowserConfig(headless=True, verbose=False)
        
        # Configure LLM
        llm_config = LLMConfig(
            provider=model_config['provider'],
            api_token=model_config.get('api_token'),
            base_url=model_config.get('base_url'),
            temperature=0.7,
            max_tokens=2000
        )
        
        # Configure extraction strategy
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            extraction_type="schema",
            schema=schema.model_json_schema(),
            instruction=instruction,
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
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.success:
                try:
                    extracted_data = json.loads(result.extracted_content) if result.extracted_content else {}
                except json.JSONDecodeError:
                    extracted_data = {"raw_content": result.extracted_content}
                
                return {
                    "model": model_config['name'],
                    "test_name": test_name,
                    "success": True,
                    "execution_time": execution_time,
                    "markdown_length": len(result.markdown.raw_markdown),
                    "extracted_data": extracted_data,
                    "token_usage": getattr(result, 'token_usage', None),
                    "error": None
                }
            else:
                return {
                    "model": model_config['name'],
                    "test_name": test_name,
                    "success": False,
                    "execution_time": execution_time,
                    "error": result.error_message,
                    "extracted_data": None
                }
                
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "model": model_config['name'],
            "test_name": test_name,
            "success": False,
            "execution_time": execution_time,
            "error": str(e),
            "extracted_data": None
        }


async def compare_news_extraction():
    """Compare both models on news extraction task"""
    print("\n📰 Testing News Extraction")
    
    url = "https://www.reuters.com/technology/"
    instruction = """
    Extract news information from this technology news page. 
    Identify the main headlines, topics being covered, and provide a summary.
    Focus on the most prominent and recent news items.
    """
    
    # Model configurations
    models = [
        {
            "name": "Gemini 2.0 Flash",
            "provider": "gemini/gemini-2.0-flash",
            "api_token": os.getenv("GEMINI_API_KEY")
        },
        {
            "name": "Devstral Small 2505",
            "provider": "ollama/devstral-small-2505",
            "api_token": "no-token-needed",
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    ]
    
    results = []
    for model_config in models:
        result = await run_extraction_test(
            model_config=model_config,
            url=url,
            schema=NewsExtraction,
            instruction=instruction,
            test_name="News Extraction"
        )
        results.append(result)
    
    return results


async def compare_tech_analysis():
    """Compare both models on technical content analysis"""
    print("\n💻 Testing Technical Content Analysis")
    
    url = "https://docs.docker.com/get-started/"
    instruction = """
    Analyze this technical documentation for programming and technology content.
    Identify technologies, tools, concepts discussed, assess the complexity level,
    and provide a comprehensive summary suitable for developers.
    """
    
    # Model configurations
    models = [
        {
            "name": "Gemini 1.5 Pro",
            "provider": "gemini/gemini-1.5-pro",
            "api_token": os.getenv("GEMINI_API_KEY")
        },
        {
            "name": "Devstral Small 2505",
            "provider": "ollama/devstral-small-2505",
            "api_token": "no-token-needed",
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    ]
    
    results = []
    for model_config in models:
        result = await run_extraction_test(
            model_config=model_config,
            url=url,
            schema=TechContentAnalysis,
            instruction=instruction,
            test_name="Technical Analysis"
        )
        results.append(result)
    
    return results


def analyze_results(all_results: List[Dict[str, Any]]):
    """Analyze and compare results from both models"""
    print("\n" + "="*70)
    print("📊 DETAILED COMPARISON ANALYSIS")
    print("="*70)
    
    # Group results by test
    tests = {}
    for result in all_results:
        test_name = result['test_name']
        if test_name not in tests:
            tests[test_name] = []
        tests[test_name].append(result)
    
    # Analyze each test
    for test_name, test_results in tests.items():
        print(f"\n🔍 {test_name} Comparison:")
        print("-" * 50)
        
        successful_results = [r for r in test_results if r['success']]
        failed_results = [r for r in test_results if not r['success']]
        
        if successful_results:
            print("✅ Successful Models:")
            
            # Compare execution times
            for result in successful_results:
                print(f"  📍 {result['model']}:")
                print(f"    ⏱️  Execution time: {result['execution_time']:.2f}s")
                print(f"    📄 Markdown length: {result['markdown_length']} chars")
                
                # Show extracted data preview
                extracted = result['extracted_data']
                if extracted:
                    print(f"    📊 Extracted fields: {list(extracted.keys()) if isinstance(extracted, dict) else 'Raw content'}")
                    
                    # Show some sample data
                    if isinstance(extracted, dict):
                        for key, value in list(extracted.items())[:2]:  # Show first 2 fields
                            if isinstance(value, list):
                                print(f"    📝 {key}: {len(value)} items - {value[:2] if value else '[]'}")
                            elif isinstance(value, str):
                                preview = value[:100] + "..." if len(value) > 100 else value
                                print(f"    📝 {key}: {preview}")
                print()
            
            # Performance comparison
            if len(successful_results) > 1:
                fastest = min(successful_results, key=lambda x: x['execution_time'])
                slowest = max(successful_results, key=lambda x: x['execution_time'])
                
                print(f"🏆 Fastest: {fastest['model']} ({fastest['execution_time']:.2f}s)")
                print(f"🐌 Slowest: {slowest['model']} ({slowest['execution_time']:.2f}s)")
                
                speed_diff = slowest['execution_time'] - fastest['execution_time']
                print(f"⚡ Speed difference: {speed_diff:.2f}s ({(speed_diff/fastest['execution_time']*100):.1f}% faster)")
        
        if failed_results:
            print("\n❌ Failed Models:")
            for result in failed_results:
                print(f"  - {result['model']}: {result['error']}")
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"  Total tests run: {len(all_results)}")
    print(f"  Successful: {len([r for r in all_results if r['success']])}")
    print(f"  Failed: {len([r for r in all_results if not r['success']])}")
    
    # Model-specific success rates
    model_stats = {}
    for result in all_results:
        model = result['model']
        if model not in model_stats:
            model_stats[model] = {'total': 0, 'success': 0}
        model_stats[model]['total'] += 1
        if result['success']:
            model_stats[model]['success'] += 1
    
    print(f"\n🎯 Model Success Rates:")
    for model, stats in model_stats.items():
        rate = (stats['success'] / stats['total']) * 100
        print(f"  {model}: {stats['success']}/{stats['total']} ({rate:.1f}%)")


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔧 Checking Prerequisites")
    
    issues = []
    
    # Check Gemini API key
    if not os.getenv("GEMINI_API_KEY"):
        issues.append("GEMINI_API_KEY environment variable not set")
    else:
        print("✅ Gemini API key is set")
    
    # Check Ollama URL
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if not ollama_url:
        print("⚠️ OLLAMA_BASE_URL not set, using default: http://localhost:11434")
    else:
        print(f"✅ Ollama URL is set: {ollama_url}")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


async def main():
    """Main comparison function"""
    print("🚀 Starting LLM Model Comparison: Gemini vs Devstral")
    print("=" * 70)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease fix the issues above before running the comparison.")
        return
    
    # Store all results
    all_results = []
    
    try:
        # Run comparison tests
        news_results = await compare_news_extraction()
        all_results.extend(news_results)
        
        tech_results = await compare_tech_analysis()
        all_results.extend(tech_results)
        
        # Analyze and compare results
        analyze_results(all_results)
        
        # Save detailed results
        timestamp = int(time.time())
        filename = f"llm_comparison_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {filename}")
        
        # Generate summary report
        summary = {
            "timestamp": timestamp,
            "total_tests": len(all_results),
            "successful_tests": len([r for r in all_results if r['success']]),
            "failed_tests": len([r for r in all_results if not r['success']]),
            "models_tested": list(set([r['model'] for r in all_results])),
            "tests_performed": list(set([r['test_name'] for r in all_results])),
            "average_execution_times": {}
        }
        
        # Calculate average execution times per model
        for result in all_results:
            if result['success']:
                model = result['model']
                if model not in summary['average_execution_times']:
                    summary['average_execution_times'][model] = []
                summary['average_execution_times'][model].append(result['execution_time'])
        
        for model, times in summary['average_execution_times'].items():
            summary['average_execution_times'][model] = sum(times) / len(times)
        
        summary_filename = f"comparison_summary_{timestamp}.json"
        with open(summary_filename, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary report saved to: {summary_filename}")
        
    except Exception as e:
        print(f"\n💥 Comparison failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())