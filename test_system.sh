#!/bin/bash
# Football Analytics System Test Script
# This script tests all major functionality components

set -e  # Exit on any error

echo "🚀 Football Analytics System Test Suite"
echo "========================================"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -e "\n${YELLOW}Testing: $test_name${NC}"
    echo "Command: $test_command"

    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((TESTS_FAILED++))
    fi
}

echo -e "\n1. Testing Basic CLI Functionality..."
run_test "CLI Help" "python3 cli.py --help > /dev/null"

echo -e "\n2. Testing Double Chance Markets..."
run_test "DC Markets Dry Run" "python3 cli.py --task full-league --leagues E1 --enable-double-chance --dc-min-prob 0.75 --fixtures-date 20251129 --dry-run --verbose"

echo -e "\n3. Testing ML Integration..."
run_test "ML Predict Mode Dry Run" "python3 cli.py --task full-league --leagues E1 --ml-mode predict --ml-algorithms rf --fixtures-date 20251129 --dry-run --verbose"

echo -e "\n4. Testing Parallel Processing Arguments..."
run_test "Parallel Workers Argument" "python3 cli.py --task full-league --leagues E1,E2 --parallel-workers 2 --fixtures-date 20251129 --dry-run"

echo -e "\n5. Testing Corner Analysis..."
run_test "Corner Analysis Dry Run" "python3 cli.py --task corners --leagues E1 --dry-run --verbose"

echo -e "\n6. Testing Data Conversion Scripts..."
if [ -f "data/todays_fixtures_20251129.json" ]; then
    run_test "Convert Upcoming Matches" "python3 src/convert_upcoming_matches.py --input data/todays_fixtures_20251129.json --output-dir /tmp --date 2025-11-29 --dry-run"
else
    echo -e "${YELLOW}⚠️  SKIPPED: Convert Upcoming Matches (fixture file not found)${NC}"
fi

echo -e "\n7. Testing Dynamic League Processing..."
if [ -f "scripts/run_selected_competitions.py" ]; then
    run_test "Dynamic League Script" "python3 scripts/run_selected_competitions.py --help > /dev/null"
else
    echo -e "${YELLOW}⚠️  SKIPPED: Dynamic League Processing (script not found)${NC}"
fi

echo -e "\n8. Testing File Structure..."
run_test "Required Directories Exist" "test -d src && test -d data && test -d ReadMeDocs"

echo -e "\n9. Testing Import Dependencies..."
run_test "Python Dependencies" "python3 -c 'import pandas, numpy, json, argparse, logging; print(\"Dependencies OK\")'"

# Check if recent output files exist
echo -e "\n10. Checking Recent Output Files..."
LATEST_ANALYSIS=$(find data/analysis -name "full_league_suggestions_*.json" -type f -mtime -7 2>/dev/null | head -1)
if [ -n "$LATEST_ANALYSIS" ]; then
    echo -e "${GREEN}✅ Found recent analysis: $(basename "$LATEST_ANALYSIS")${NC}"
    ((TESTS_PASSED++))

    # Verify JSON structure
    if python3 -c "import json; json.load(open('$LATEST_ANALYSIS'))" 2>/dev/null; then
        echo -e "${GREEN}✅ JSON structure valid${NC}"
        ((TESTS_PASSED++))

        # Check for DC markets in output
        if python3 -c "
import json
with open('$LATEST_ANALYSIS') as f:
    data = json.load(f)
    has_dc = any('DC' in suggestion.get('markets', {}) for suggestion in data.get('suggestions', []))
    has_ml = any('ml_prediction' in suggestion for suggestion in data.get('suggestions', []))
    print(f'DC Markets: {has_dc}, ML Predictions: {has_ml}')
    assert has_dc and has_ml, 'Missing DC or ML functionality'
"; then
            echo -e "${GREEN}✅ DC Markets and ML predictions found in output${NC}"
            ((TESTS_PASSED++))
        else
            echo -e "${RED}❌ DC Markets or ML predictions missing in output${NC}"
            ((TESTS_FAILED++))
        fi
    else
        echo -e "${RED}❌ Invalid JSON structure${NC}"
        ((TESTS_FAILED++))
    fi
else
    echo -e "${YELLOW}⚠️  No recent analysis files found${NC}"
fi

# Performance test (optional)
if [ "$1" = "--performance" ]; then
    echo -e "\n11. Performance Test (Sequential vs Parallel)..."

    echo "Testing sequential processing..."
    TIME_SEQUENTIAL=$(time (python3 cli.py --task full-league --leagues E1,E2 --parallel-workers 1 --ml-mode predict --fixtures-date 20251129 --dry-run) 2>&1 | grep real | awk '{print $2}')

    echo "Testing parallel processing..."
    TIME_PARALLEL=$(time (python3 cli.py --task full-league --leagues E1,E2 --parallel-workers 2 --ml-mode predict --fixtures-date 20251129 --dry-run) 2>&1 | grep real | awk '{print $2}')

    echo "Sequential: $TIME_SEQUENTIAL, Parallel: $TIME_PARALLEL"
fi

# Summary
echo -e "\n========================================"
echo -e "🎯 TEST SUMMARY"
echo -e "========================================"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🚀 ALL TESTS PASSED! System is fully operational.${NC}"
    exit 0
else
    echo -e "\n${RED}⚠️  Some tests failed. Check the output above for details.${NC}"
    exit 1
fi
