# ELI API Retry Strategy for 503 Service Unavailable

## Problem Statement
The ELI API endpoint `https://api.sejm.gov.pl/eli/acts/DU/2025/` returns `503 Bad Gateway` errors, requiring intelligent handling for different operational modes.

## Solution Overview
Implement mode-based retry strategies with different levels of persistence:

### 🔄 NORMAL Mode (Automated Hourly Checks)
- **Purpose**: Regular monitoring without resource waste
- **Max retries**: 3 attempts
- **Backoff**: Exponential (1s → 2s → 4s)
- **Max wait**: 5 minutes per retry
- **Total time**: ~7 seconds maximum
- **Behavior**: Log warning, proceed to next scheduled run
- **Use case**: `uv run python sejm-whiz-cli.py ingest documents --since 1h`

### 🚨 URGENT Mode (Manual Critical Processing)
- **Purpose**: Aggressive retries when timely processing is critical
- **Max retries**: 8 attempts  
- **Backoff**: Moderate exponential (1s → 1.5s → 2.25s → 3.4s...)
- **Max wait**: 20 minutes per retry
- **Total timeout**: 1 hour
- **Behavior**: Persistent retrying with status updates
- **Use case**: `uv run python sejm-whiz-cli.py ingest documents --urgent --since 1h`

## Technical Implementation

### 1. Enhanced Error Classes
```python
class EliServiceUnavailableError(EliApiError):
    """Service temporarily unavailable (503, 502, 504)."""
    pass

class OperationalMode(Enum):
    NORMAL = "normal"   # Hourly checks, gentle retries
    URGENT = "urgent"   # Manual trigger, aggressive retries
```

### 2. Configuration Parameters
```python
@dataclass
class EliApiConfig:
    # Normal mode (default)
    normal_mode_max_retries: int = 3
    normal_mode_backoff_factor: float = 2.0
    normal_mode_max_wait: int = 300  # 5 minutes max wait
    
    # Urgent mode
    urgent_mode_max_retries: int = 8
    urgent_mode_backoff_factor: float = 1.5
    urgent_mode_max_wait: int = 1200  # 20 minutes max wait
    urgent_mode_total_timeout: int = 3600  # 1 hour total
```

### 3. Enhanced Request Method
```python
async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None):
    """Make request with intelligent retry strategy based on operational mode."""
    
    # Handle different HTTP status codes:
    # - 502, 503, 504: Apply retry strategy
    # - 429: Respect Retry-After header
    # - 404: Fail immediately (no retry)
    # - Other errors: Standard exponential backoff
    
    max_retries, backoff_factor, max_wait, total_timeout = self._get_retry_config()
    
    for attempt in range(max_retries):
        try:
            response = await self._client.get(url, params=params)
            
            # Service unavailable - apply retry strategy
            if response.status_code in [502, 503, 504]:
                wait_time = min(backoff_factor ** attempt, max_wait)
                logger.warning(f"Service unavailable ({response.status_code}), "
                              f"attempt {attempt + 1}/{max_retries}, waiting {wait_time:.1f}s")
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise EliServiceUnavailableError(f"Service unavailable after {max_retries} attempts")
            
            # Success case
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Handle specific error codes with appropriate retry logic
```

## CLI Integration

### Command Usage
```bash
# Normal mode (default) - gentle retries
uv run python sejm-whiz-cli.py ingest documents --source eli --since 1h

# Urgent mode - aggressive retries  
uv run python sejm-whiz-cli.py ingest documents --urgent --source eli --since 1h

# Check API health
uv run python sejm-whiz-cli.py system status --api-health
```

### Enhanced Logging Output
```
🚨 Operational mode: URGENT (aggressive retries)
🚀 Starting document ingestion from eli
⚠️  Service unavailable (503), attempt 1/8, waiting 1.0s
⚠️  Service unavailable (503), attempt 2/8, waiting 1.5s  
⚠️  Service unavailable (503), attempt 3/8, waiting 2.3s
✅ Successfully fetched 1076 documents from DU/2025 (URL: https://api.sejm.gov.pl/eli/acts/DU/2025/)
```

## Error Handling Matrix

| HTTP Status | Normal Mode | Urgent Mode | Action |
|-------------|-------------|-------------|---------|
| 200 OK | ✅ Process | ✅ Process | Continue normally |
| 404 Not Found | ❌ Fail fast | ❌ Fail fast | No retry (document doesn't exist) |
| 429 Rate Limited | ⏳ Respect Retry-After | ⏳ Respect Retry-After | Wait as instructed by API |
| 502 Bad Gateway | 🔄 3 retries (7s total) | 🔄 8 retries (up to 1h) | Apply retry strategy |
| 503 Service Unavailable | 🔄 3 retries (7s total) | 🔄 8 retries (up to 1h) | Apply retry strategy |
| 504 Gateway Timeout | 🔄 3 retries (7s total) | 🔄 8 retries (up to 1h) | Apply retry strategy |

## Monitoring and Alerting

### Metrics to Track
- API availability percentage
- Retry attempt counts by mode
- Success rate after retries
- Average response times
- Service outage duration

### Alerting Rules
- **Normal Mode**: Log warnings, continue operation
- **Urgent Mode**: 
  - Send notification after 3 failed attempts
  - Escalate alert after 15 minutes of failures
  - Critical alert if 1-hour timeout reached

### Dashboard Indicators
- 🟢 API Healthy (< 5% errors in last hour)
- 🟡 API Degraded (5-20% errors, retries succeeding)
- 🔴 API Down (> 20% errors or extended outage)

## Fallback Strategies

### When API is Completely Unavailable
1. **Cache**: Serve from recent cached data if available
2. **Alternative Endpoints**: Try different API paths if configured
3. **Graceful Degradation**: Continue with partial data
4. **User Notification**: Clear status messages about API availability

### Circuit Breaker Pattern
- Open circuit after 10 consecutive failures
- Half-open after 5-minute cool-down
- Close circuit after 3 consecutive successes

## Benefits

### For Normal Mode (Automated)
- ✅ Resource efficient 
- ✅ Prevents cascading failures
- ✅ Maintains system stability
- ✅ Suitable for scheduled tasks

### For Urgent Mode (Manual)
- ✅ Higher success rate for critical operations
- ✅ Comprehensive retry strategy
- ✅ Detailed progress feedback
- ✅ Suitable for breaking news, urgent legal changes

## Usage Examples

### Scenario 1: Regular Monitoring
```bash
# Cron job running every hour
0 * * * * /usr/bin/uv run python /path/to/sejm-whiz-cli.py ingest documents --since 1h
```

### Scenario 2: Breaking Legal News
```bash
# Manual trigger with aggressive retries
uv run python sejm-whiz-cli.py ingest documents --urgent --since 24h --limit 100
```

### Scenario 3: API Health Check
```bash
# Monitor API status
uv run python sejm-whiz-cli.py system status --api-health --json
```

This strategy ensures robust handling of API outages while providing appropriate behavior for different use cases.