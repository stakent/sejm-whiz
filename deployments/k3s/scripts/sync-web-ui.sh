#!/bin/bash
# Fast sync script for web UI development
# Syncs only web UI files to p7 for instant hot reload

set -e

echo "🔄 Syncing Web UI files to p7 for hot reload..."

# Check if we're in the right directory
if [[ ! -f "workspace.toml" ]]; then
    echo "❌ Error: Please run this script from the sejm-whiz project root"
    exit 1
fi

# Sync only necessary files for web UI (much faster than full project sync)
echo "📁 Syncing web UI project files..."
rsync -avz --delete \
    --include="projects/" \
    --include="projects/web_ui/" \
    --include="projects/web_ui/**" \
    --include="components/" \
    --include="components/sejm_whiz/" \
    --include="components/sejm_whiz/web_api/" \
    --include="components/sejm_whiz/web_api/**" \
    --include="bases/" \
    --include="bases/web_api/" \
    --include="bases/web_api/**" \
    --exclude="*" \
    . root@p7:/tmp/sejm-whiz/

echo "✅ Files synced! Changes should be reflected immediately."
echo ""
echo "📋 Access URLs:"
echo "  - Production URL: http://192.168.0.200:30800/"
echo "  - Development URL: http://192.168.0.200:30801/"
echo ""
echo "📋 Quick Commands:"
echo "  - View logs: ssh root@p7 'kubectl logs -n sejm-whiz deployment/sejm-whiz-web-ui-dev -f'"
echo "  - Check status: ssh root@p7 'kubectl get pods -n sejm-whiz -l app=sejm-whiz-web-ui-dev'"
echo ""
echo "💡 Tip: Run this script after making changes to see them instantly!"
echo ""
echo "⚠️  Known Issues (for refinement):"
echo "  - Web UI may show deployment issues in browser (needs volume mount adjustments)"
echo "  - Error monitoring stream may not load properly in browser"
echo "  - Production service pointing to dev deployment as temporary fix"