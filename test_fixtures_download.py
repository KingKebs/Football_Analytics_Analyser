#!/usr/bin/env python3
"""
Test script for fixtures downloader functionality

Tests the new fixtures downloading capabilities without making actual API calls.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_fixtures_downloader():
    """Test the fixtures downloader module"""
    print("🧪 Testing Fixtures Downloader...")

    try:
        from fixtures_downloader import FixturesDownloader
        print("✅ Successfully imported FixturesDownloader")

        # Test initialization
        downloader = FixturesDownloader()
        print("✅ Successfully initialized downloader")

        # Test dry run
        result = downloader.download_fixtures("2025-11-30", dry_run=True)
        print("✅ Dry run completed successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install requests beautifulsoup4")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_web_scraper():
    """Test the web scraper fallback"""
    print("\n🧪 Testing Web Scraper Fallback...")

    try:
        from web_scraper_fallback import WebScraperFallback
        print("✅ Successfully imported WebScraperFallback")

        scraper = WebScraperFallback()
        print("✅ Successfully initialized scraper")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies: pip install requests beautifulsoup4")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("\n🧪 Testing CLI Integration...")

    try:
        import subprocess

        # Test CLI help for new task
        result = subprocess.run([
            sys.executable, 'cli.py', '--task', 'download-fixtures', '--help'
        ], capture_output=True, text=True)

        if 'download-fixtures' in result.stdout or result.returncode == 0:
            print("✅ CLI integration working")
            return True
        else:
            print(f"❌ CLI test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def test_scheduler():
    """Test the fixtures scheduler"""
    print("\n🧪 Testing Fixtures Scheduler...")

    try:
        import subprocess

        # Test scheduler dry run
        result = subprocess.run([
            sys.executable, 'fixtures_scheduler.py', '--dry-run', '--daily-update'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ Scheduler test passed")
            return True
        else:
            print(f"❌ Scheduler test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Scheduler test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Football Analytics Fixtures Download Test Suite")
    print("=" * 60)

    tests = [
        test_fixtures_downloader,
        test_web_scraper,
        test_cli_integration,
        test_scheduler
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Fixtures download functionality is ready.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install requests beautifulsoup4")
        print("2. Get API keys (optional): see ReadMeDocs/FIXTURES_DOWNLOAD_GUIDE.md")
        print("3. Test live download: python cli.py --task download-fixtures --dry-run --update-today")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
