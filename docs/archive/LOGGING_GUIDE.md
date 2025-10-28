# 📊 Enterprise Logging Guide

**⏱️ 10 min read | Learn production-grade logging**

---

## 🎯 Why Enterprise Logging Matters

**Bad logging:**
```python
print("starting search")
print("error happened")
print(f"got {len(results)} results")
```

**Problems:**
- Can't search logs later
- No timestamps
- No context
- Lost when app restarts
- Can't filter by severity

**Enterprise logging:**
```python
logger.info("Search initiated", extra={"query": query, "user_id": user_id})
logger.error("API call failed", exc_info=True, extra={"api": "openai", "status_code": 500})
logger.debug(f"Retrieved {len(results)} results", extra={"result_count": len(results)})
```

**Benefits:**
- ✅ Searchable (find specific errors)
- ✅ Timestamped (know when it happened)
- ✅ Contextual (understand what was happening)
- ✅ Persistent (saved to files)
- ✅ Filterable (show only errors)
- ✅ Professional (what real companies use)

---

## 🏢 What This Project Has

### 1. **Structured Logging** (`src/logging_config.py`)
```python
# Human-readable logs
2025-10-10 14:30:15 - websearch.main - INFO - Search initiated

# JSON logs (for tools like ELK, Splunk, DataDog)
{
  "timestamp": "2025-10-10T14:30:15Z",
  "level": "INFO",
  "logger": "websearch.main",
  "message": "Search initiated",
  "query": "AI news"
}
```

### 2. **Log Rotation**
```
logs/
├── app.log          # Current log (max 10MB)
├── app.log.1        # Previous log
├── app.log.2        # Older log
├── app.log.3        # Older log
├── app.log.4        # Older log
└── error.log        # Only errors
```

**Why?** Prevents disk from filling up. Keeps last 5 files (50MB total).

### 3. **Log Levels**
| Level | When to Use | Example |
|-------|-------------|---------|
| **DEBUG** | Detailed info for debugging | `logger.debug(f"API payload: {payload}")` |
| **INFO** | General informational messages | `logger.info("Search completed successfully")` |
| **WARNING** | Something unexpected (but not error) | `logger.warning("Rate limit approaching")` |
| **ERROR** | Error occurred, but app continues | `logger.error("API call failed", exc_info=True)` |
| **CRITICAL** | Serious error, app might crash | `logger.critical("Database unreachable")` |

### 4. **Context Information**
Every log includes:
- Timestamp (when)
- Module (where)
- Function (which function)
- Line number (exact line)
- Message (what happened)
- Extra fields (additional context)

---

## 🚀 How To Use Logging

### Basic Usage

```python
from src.logging_config import get_logger

# Get logger for your module
logger = get_logger(__name__)

# Log messages
logger.debug("This is debugging info")
logger.info("This is informational")
logger.warning("This is a warning")
logger.error("This is an error")
logger.critical("This is critical!")
```

### Logging with Context

```python
# Add extra context
logger.info(
    "Search completed",
    extra={
        "query": "AI news",
        "result_count": 10,
        "duration_ms": 250
    }
)

# Output (JSON format):
{
  "timestamp": "2025-10-10T14:30:15Z",
  "level": "INFO",
  "message": "Search completed",
  "query": "AI news",
  "result_count": 10,
  "duration_ms": 250
}
```

### Logging Exceptions

```python
try:
    result = api.search(query)
except Exception as e:
    # exc_info=True includes full stack trace
    logger.error(
        f"Search failed: {e}",
        exc_info=True,
        extra={"query": query}
    )
    raise
```

### Performance Logging

```python
from src.logging_config import log_performance, get_logger

logger = get_logger(__name__)

@log_performance(logger)
def expensive_operation():
    # This will automatically log execution time
    process_data()

# Output:
# INFO - expensive_operation completed successfully (duration_ms: 523.45)
```

### Context Manager

```python
from src.logging_config import LogContext, get_logger

logger = get_logger(__name__)

with LogContext(logger, "Processing search", query="AI news"):
    # Everything in this block is timed and logged
    results = search_api(query)
    parsed = parse_results(results)

# Output:
# INFO - Starting: Processing search
# INFO - Completed: Processing search (duration_ms: 234.56)
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Set log level (DEBUG shows everything)
LOG_LEVEL=INFO

# Where to save logs
LOG_DIR=logs

# Format: text or json
LOG_FORMAT=text  # Use 'json' for production
```

### Log Levels Explained

**Development:**
```bash
LOG_LEVEL=DEBUG  # See everything
LOG_FORMAT=text  # Easy to read
```

**Production:**
```bash
LOG_LEVEL=INFO   # Normal operations
LOG_FORMAT=json  # Easy to parse with tools
```

**Troubleshooting:**
```bash
LOG_LEVEL=DEBUG  # Maximum detail
LOG_FORMAT=text  # Human-readable
```

---

## 📋 Best Practices

### ✅ DO:

**1. Log Important Events**
```python
logger.info("User authenticated", extra={"user_id": user_id})
logger.info("Search initiated", extra={"query": query})
logger.info("Results returned", extra={"count": len(results)})
```

**2. Log Errors with Context**
```python
logger.error(
    "API call failed",
    exc_info=True,  # Include stack trace
    extra={
        "api": "openai",
        "query": query,
        "status_code": response.status_code
    }
)
```

**3. Use Appropriate Levels**
```python
logger.debug("Cache hit for query")      # DEBUG: Detailed info
logger.info("Search completed")           # INFO: Normal operation
logger.warning("Cache nearly full")       # WARNING: Potential issue
logger.error("API rate limit exceeded")   # ERROR: Problem occurred
logger.critical("Database unreachable")   # CRITICAL: Serious problem
```

**4. Include Context**
```python
# Good - includes context
logger.info("Search completed", extra={
    "query": query,
    "result_count": len(results),
    "duration_ms": duration
})

# Bad - no context
logger.info("Search completed")
```

### ❌ DON'T:

**1. Don't Log Sensitive Data**
```python
# BAD - logs API key!
logger.info(f"Using API key: {api_key}")

# GOOD - masks sensitive data
logger.info(f"Using API key: {api_key[:8]}...")
```

**2. Don't Log Inside Loops**
```python
# BAD - creates 1000 log entries
for item in items:
    logger.info(f"Processing {item}")

# GOOD - log summary
logger.info(f"Processing {len(items)} items")
```

**3. Don't Use print() Statements**
```python
# BAD
print("Search started")

# GOOD
logger.info("Search started")
```

**4. Don't Log Excessively**
```python
# BAD - too verbose
logger.debug("Entering function")
logger.debug("Variable x = 5")
logger.debug("Variable y = 10")
logger.debug("Returning result")

# GOOD - log meaningful events
logger.debug(f"Calculating result with x={x}, y={y}")
```

---

## 🔍 Viewing Logs

### Console Output
```bash
# Run your app - see logs in console
python -m src.main "AI news"

# Output:
2025-10-10 14:30:15 - websearch.main - INFO - Search initiated
2025-10-10 14:30:16 - websearch.client - INFO - API call successful
2025-10-10 14:30:16 - websearch.main - INFO - Search completed
```

### File Logs
```bash
# View application log
tail -f logs/app.log

# View only errors
tail -f logs/error.log

# Search logs for specific query
grep "AI news" logs/app.log

# View last 100 lines
tail -100 logs/app.log

# Follow logs in real-time
tail -f logs/app.log | grep ERROR
```

### JSON Logs (Production)
```bash
# Enable JSON logging
echo "LOG_FORMAT=json" >> .env

# Query JSON logs with jq
cat logs/app.log | jq '.message'
cat logs/app.log | jq 'select(.level == "ERROR")'
cat logs/app.log | jq 'select(.duration_ms > 1000)'
```

---

## 🎓 For Your Project

### Step 1: Import Logging
```python
from src.logging_config import get_logger

logger = get_logger(__name__)
```

### Step 2: Add Logs at Key Points
```python
def my_function(data):
    logger.info("Function started", extra={"data_size": len(data)})
    
    try:
        result = process_data(data)
        logger.info("Processing successful", extra={"result_size": len(result)})
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

### Step 3: Use Context Managers for Operations
```python
with LogContext(logger, "Data processing", task_id=task_id):
    # Your code here
    process_data()
    save_results()
# Automatically logs start, end, and duration
```

### Step 4: Log Performance
```python
@log_performance(logger)
def expensive_function():
    # Automatically logs execution time
    do_complex_work()
```

---

## 📊 Log Aggregation (Advanced)

In production, logs are sent to aggregation tools:

### Popular Tools:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Splunk** (Enterprise log management)
- **DataDog** (Monitoring & analytics)
- **CloudWatch** (AWS logging)
- **Grafana Loki** (Open-source)

### Our JSON Format Works With All:
```json
{
  "timestamp": "2025-10-10T14:30:15Z",
  "level": "ERROR",
  "logger": "websearch.client",
  "message": "API call failed",
  "api": "openai",
  "status_code": 429,
  "duration_ms": 1523
}
```

---

## 🎯 Logging Checklist

For each function in your project, ask:

- [ ] Does it log when it starts? (INFO level)
- [ ] Does it log when it succeeds? (INFO level)
- [ ] Does it log errors? (ERROR level with exc_info=True)
- [ ] Does it include context? (extra fields)
- [ ] Does it log performance? (duration for slow operations)
- [ ] Does it avoid logging sensitive data? (API keys, passwords)
- [ ] Does it use appropriate log levels?

---

## 💡 Real-World Example

```python
from src.logging_config import get_logger, LogContext

logger = get_logger(__name__)

def search_and_process(query: str, user_id: str):
    """Search and process results with enterprise logging."""
    
    # Log operation start
    logger.info("Search request received", extra={
        "query": query,
        "user_id": user_id
    })
    
    # Use context manager for timing
    with LogContext(logger, "Search operation", query=query, user_id=user_id):
        try:
            # Perform search
            logger.debug(f"Calling search API with query: {query}")
            results = api.search(query)
            
            # Log success with metrics
            logger.info("Search successful", extra={
                "result_count": len(results),
                "query": query
            })
            
            # Process results
            logger.debug("Processing search results")
            processed = process_results(results)
            
            logger.info("Processing complete", extra={
                "processed_count": len(processed)
            })
            
            return processed
            
        except APIError as e:
            # Log specific API errors
            logger.error(
                "API error occurred",
                exc_info=True,
                extra={
                    "error_code": e.code,
                    "query": query,
                    "user_id": user_id
                }
            )
            raise
            
        except Exception as e:
            # Log unexpected errors
            logger.critical(
                "Unexpected error in search",
                exc_info=True,
                extra={"query": query, "user_id": user_id}
            )
            raise
```

---

## 🚀 Next Steps

1. ✅ Read this guide
2. ➡️ Look at `src/logging_config.py` to understand implementation
3. ➡️ Check `src/main.py` for real examples
4. ➡️ Add logging to your own project functions
5. ➡️ Run your app and check `logs/app.log`

---

## 📖 Vocabulary for Your Resume

Now you can say:
- ✅ "Implemented structured logging with log rotation"
- ✅ "Used JSON logging for log aggregation"
- ✅ "Configured multi-level logging (DEBUG, INFO, ERROR)"
- ✅ "Implemented performance logging with context managers"
- ✅ "Separated application and error logs"
- ✅ "Used enterprise logging patterns (ELK-compatible)"

**This is what professional developers do.** 🏢

---

*Need help adding logging to your code? Ask Claude: "Show me how to add enterprise logging to this function: [paste code]"*
