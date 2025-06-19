#!/usr/bin/env python3
"""
Environment setup script for LLM testing with Crawl4AI
This script helps configure the environment for testing Gemini and Devstral models.
"""

import os
import sys
from pathlib import Path


def create_env_template():
    """Create a template .env file for the tests"""
    env_content = """# Environment variables for Crawl4AI LLM testing

# Google Gemini API Configuration
# Get your API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration for Devstral
# Set this to your remote Ollama server URL
# Default: http://localhost:11434
OLLAMA_BASE_URL=http://your-remote-server:11434

# Optional: Other LLM providers
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
"""
    
    env_file = Path(".env.template")
    env_file.write_text(env_content)
    print(f"✅ Created environment template: {env_file}")
    print("Please copy this to .env and fill in your actual API keys and URLs")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'crawl4ai',
        'pydantic',
        'httpx',
        'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def verify_crawl4ai_installation():
    """Verify Crawl4AI is properly installed"""
    try:
        from crawl4ai import AsyncWebCrawler
        print("✅ Crawl4AI imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Crawl4AI: {e}")
        print("Install with: pip install crawl4ai")
        return False


def check_environment_variables():
    """Check current environment variables"""
    print("\n🔍 Checking Environment Variables:")
    
    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...{gemini_key[-4:] if len(gemini_key) > 14 else '[short]'}")
    else:
        print("⚠️  GEMINI_API_KEY: Not set")
    
    # Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_url:
        print(f"✅ OLLAMA_BASE_URL: {ollama_url}")
    else:
        print("⚠️  OLLAMA_BASE_URL: Not set (will use default: http://localhost:11434)")
    
    return bool(gemini_key)  # At minimum, we need Gemini key set


def provide_setup_instructions():
    """Provide setup instructions"""
    print("\n📋 Setup Instructions:")
    print("=" * 50)
    
    print("\n1. 🔑 Get Google Gemini API Key:")
    print("   - Visit: https://aistudio.google.com/app/apikey")
    print("   - Create a new API key")
    print("   - Set it: export GEMINI_API_KEY='your_key_here'")
    
    print("\n2. 🖥️  Setup Ollama with Devstral:")
    print("   - Install Ollama on your remote server")
    print("   - Pull Devstral model: ollama pull devstral-small-2505")
    print("   - Set URL: export OLLAMA_BASE_URL='http://your-server:11434'")
    
    print("\n3. 📦 Install Dependencies:")
    print("   - pip install crawl4ai")
    print("   - pip install pydantic httpx")
    
    print("\n4. 🧪 Run Tests:")
    print("   - Test Gemini: python test_gemini_integration.py")
    print("   - Test Devstral: python test_devstral_integration.py")
    print("   - Compare both: python test_llm_comparison.py")


def test_basic_connectivity():
    """Test basic connectivity to services"""
    print("\n🔗 Testing Connectivity:")
    
    # Test Gemini (simple import test)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("✅ Gemini API key is available for testing")
    else:
        print("⚠️  Gemini API key not available")
    
    # Test Ollama connectivity
    import asyncio
    import httpx
    
    async def test_ollama():
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    print(f"✅ Ollama server accessible at {ollama_url}")
                    models = response.json().get("models", [])
                    devstral_models = [m for m in models if "devstral" in m.get("name", "").lower()]
                    if devstral_models:
                        print(f"✅ Devstral model(s) found: {[m['name'] for m in devstral_models]}")
                    else:
                        print("⚠️  No Devstral models found. Run: ollama pull devstral-small-2505")
                else:
                    print(f"❌ Ollama server returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot reach Ollama server: {e}")
    
    try:
        asyncio.run(test_ollama())
    except Exception as e:
        print(f"❌ Error testing Ollama connectivity: {e}")


def main():
    """Main setup function"""
    print("🚀 Crawl4AI LLM Testing Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    print("\n📦 Checking Dependencies:")
    if not check_dependencies():
        print("\nPlease install missing dependencies before proceeding.")
        return
    
    # Verify Crawl4AI
    if not verify_crawl4ai_installation():
        return
    
    # Check environment
    env_ok = check_environment_variables()
    
    # Test connectivity
    test_basic_connectivity()
    
    # Create template if needed
    if not Path(".env").exists() and not Path(".env.template").exists():
        create_env_template()
    
    # Provide instructions
    provide_setup_instructions()
    
    # Final status
    print("\n" + "=" * 50)
    if env_ok:
        print("🎉 Environment setup looks good! You can now run the tests.")
        print("\n🧪 Recommended test order:")
        print("1. python test_gemini_integration.py")
        print("2. python test_devstral_integration.py")
        print("3. python test_llm_comparison.py")
    else:
        print("⚠️  Please complete the setup steps above before running tests.")
        print("💡 Tip: Create a .env file with your API keys and URLs")


if __name__ == "__main__":
    main()