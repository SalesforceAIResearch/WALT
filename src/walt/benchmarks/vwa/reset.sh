#!/bin/bash
# Reset benchmark environments
# Usage: ./src/walt/benchmarks/scripts/reset_benchmark.sh [domain]
#   domain: classifieds, shopping, reddit, gitlab (default: all)

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DOMAIN=${1:-all}

echo "════════════════════════════════════════════════════════════════"
echo "  🔄 Benchmark Reset Script"
echo "════════════════════════════════════════════════════════════════"
echo ""

reset_classifieds() {
    echo "Resetting Classifieds..."
    if [ -z "$CLASSIFIEDS" ] || [ -z "$CLASSIFIEDS_RESET_TOKEN" ]; then
        echo "❌ CLASSIFIEDS or CLASSIFIEDS_RESET_TOKEN not set in .env"
        return 1
    fi
    
    curl -X POST "${CLASSIFIEDS}/index.php?page=reset" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "token=${CLASSIFIEDS_RESET_TOKEN}" \
        > /dev/null 2>&1
    
    echo "✅ Classifieds reset"
}

reset_shopping() {
    echo "Resetting Shopping..."
    if [ -z "$SHOPPING" ]; then
        echo "❌ SHOPPING not set in .env"
        return 1
    fi
    
    # Shopping reset logic (depends on your setup)
    # Add appropriate reset commands here
    echo "⚠️  Shopping reset not implemented (add your reset logic)"
}

reset_reddit() {
    echo "Resetting Reddit..."
    if [ -z "$REDDIT" ]; then
        echo "❌ REDDIT not set in .env"
        return 1
    fi
    
    # Reddit reset logic
    echo "⚠️  Reddit reset not implemented (add your reset logic)"
}

reset_gitlab() {
    echo "Resetting GitLab..."
    if [ -z "$GITLAB" ]; then
        echo "❌ GITLAB not set in .env"
        return 1
    fi
    
    # GitLab reset logic
    echo "⚠️  GitLab reset not implemented (add your reset logic)"
}

# Main
case $DOMAIN in
    classifieds)
        reset_classifieds
        ;;
    shopping)
        reset_shopping
        ;;
    reddit)
        reset_reddit
        ;;
    gitlab)
        reset_gitlab
        ;;
    all)
        reset_classifieds || true
        reset_shopping || true
        reset_reddit || true
        reset_gitlab || true
        ;;
    *)
        echo "❌ Unknown domain: $DOMAIN"
        echo "Usage: $0 [classifieds|shopping|reddit|gitlab|all]"
        exit 1
        ;;
esac

echo ""
echo "✅ Reset complete"

