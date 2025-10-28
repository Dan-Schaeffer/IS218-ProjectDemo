# 📊 Enterprise Logging - Quick Guide

**⏱️ 5 min read | Log like a pro**

---

## 🎯 Why Logging?

### ❌ Bad (Print Statements):

```python
def process_user(user_id):
    print("Processing user")
    user = get_user(user_id)
    print(f"Got user: {user}")
    result = do_something(user)
    print("Done")
    return result
```

**Problems:**
- Can't turn off prints in production
- No timestamps
- No severity levels
- Lost when program ends
- Mixed with legitimate output

### ✅ Good (Logging):

```python
import logging
logger = logging.getLogger(__name__)

def process_user(user_id):
    logger.info(f"Processing user {user_id}")
    user = get_user(user_id)
    logger.debug(f"Retrieved user: {user}")
    result = do_something(user)
    logger.info(f"Completed processing for user {user_id}")
    return result
```

**Benefits:**
- Configurable levels
- Timestamps automatic
- Saved to file
- Production-ready
- Professional

---

## 🚦 Log Levels (When to Use Each)

| Level | When to use | Example |
|-------|------------|---------|
| `DEBUG` | Detailed info for debugging | `"User object: {user}"` |
| `INFO` | General information | `"User logged in"` |
| `WARNING` | Something unexpected but handled | `"Retrying API call (3/3)"` |
| `ERROR` | Something failed but app continues | `"Failed to process user X"` |
| `CRITICAL` | Something failed, app may crash | `"Database connection lost"` |

---

## 📝 Quick Start

### Setup (Do Once):

```python
# src/main.py
from logging_config import setup_logging, get_logger

# At the start of your program
setup_logging()
logger = get_logger(__name__)

logger.info("Application started")
```

### In Other Files:

```python
# src/any_file.py
from logging_config import get_logger

logger = get_logger(__name__)

def my_function():
    logger.info("Function called")
    # Your code
    logger.debug(f"Result: {result}")
```

---

## 🎯 Common Patterns

### Pattern 1: Function Entry/Exit

```python
def search_web(query: str):
    logger.info(f"Starting web search for: {query}")
    
    results = do_search(query)
    
    logger.info(f"Search complete. Found {len(results)} results")
    return results
```

---

### Pattern 2: Error Handling

```python
def call_api(endpoint: str):
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        logger.info(f"API call successful: {endpoint}")
        return response.json()
        
    except requests.HTTPError as e:
        logger.error(f"API call failed: {endpoint}", exc_info=True)
        raise
        
    except Exception as e:
        logger.critical(f"Unexpected error: {endpoint}", exc_info=True)
        raise
```

**Note:** `exc_info=True` includes the full traceback!

---

### Pattern 3: Timed Operations

```python
from logging_config import LogContext

def process_data(data):
    with LogContext("data_processing", logger):
        # This will automatically log:
        # - When it starts
        # - How long it took
        # - If it succeeded or failed
        
        result = expensive_operation(data)
        return result
```

**Output:**
```
INFO: Starting data_processing
INFO: Completed data_processing in 2.34s
```

---

### Pattern 4: Performance Tracking

```python
from logging_config import log_performance

@log_performance(logger)
def slow_function(x, y):
    # Complex computation
    return result
```

**Output:**
```
INFO: slow_function took 1.23s
```

---

## 📁 Where Are the Logs?

### Files Created:

```
your-project/
  logs/
    app.log      # All logs (INFO and above)
    error.log    # Only errors (ERROR and above)
```

### View Logs:

```bash
# See latest logs
tail -f logs/app.log

# See errors only
tail -f logs/error.log

# Search logs
grep "error" logs/app.log

# See logs from today
grep "2024-01-15" logs/app.log
```

---

## 🎨 Log Format

### Standard Format:

```
2024-01-15 14:32:15,123 - src.client - INFO - API call successful
│                       │            │      │
│                       │            │      └─ Message
│                       │            └─ Level
│                       └─ Module name
└─ Timestamp
```

### In Code:

```python
logger.info("User login successful")
# Output: 2024-01-15 14:32:15,123 - src.auth - INFO - User login successful

logger.error("Database connection failed", exc_info=True)
# Output: 2024-01-15 14:32:16,456 - src.db - ERROR - Database connection failed
# [Full traceback here]
```

---

## 🔧 Configuration

### Environment Variables:

```bash
# .env file
LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR=logs          # Where to save logs
LOG_FORMAT=text       # text or json
```

### Log Levels Explained:

**Development:**
```bash
LOG_LEVEL=DEBUG  # See everything
```

**Production:**
```bash
LOG_LEVEL=INFO   # Normal operations + errors
```

**Troubleshooting:**
```bash
LOG_LEVEL=DEBUG  # Turn on temporarily
```

---

## 💡 Best Practices

### DO:
- ✅ Log at function boundaries (entry/exit)
- ✅ Log errors with `exc_info=True`
- ✅ Log business events ("Order placed", "User registered")
- ✅ Use appropriate log levels
- ✅ Include relevant context (user_id, order_id)

### DON'T:
- ❌ Log sensitive data (passwords, credit cards)
- ❌ Log in tight loops (10,000 times/sec)
- ❌ Use print() in production code
- ❌ Log entire objects (use selective fields)
- ❌ Forget `exc_info=True` for errors

---

## 🚨 Common Patterns

### Pattern: API Call

```python
def fetch_user(user_id: int):
    logger.info(f"Fetching user {user_id}")
    
    try:
        response = api.get(f"/users/{user_id}")
        logger.debug(f"API response: {response.status_code}")
        
        if response.status_code == 404:
            logger.warning(f"User {user_id} not found")
            return None
            
        user = response.json()
        logger.info(f"Successfully fetched user {user_id}")
        return user
        
    except Exception as e:
        logger.error(f"Failed to fetch user {user_id}", exc_info=True)
        raise
```

---

### Pattern: Processing Loop

```python
def process_items(items: list):
    logger.info(f"Processing {len(items)} items")
    
    success_count = 0
    error_count = 0
    
    for item in items:
        try:
            process_item(item)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process item {item.id}", exc_info=True)
            error_count += 1
    
    logger.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")
```

---

### Pattern: Configuration Load

```python
def load_config():
    logger.info("Loading configuration")
    
    if not os.path.exists(".env"):
        logger.warning(".env file not found, using defaults")
        return default_config()
    
    config = read_env_file()
    
    if not config.get("API_KEY"):
        logger.error("API_KEY not set in .env")
        raise ValueError("Missing API_KEY")
    
    logger.info("Configuration loaded successfully")
    return config
```

---

## 🎯 Debugging with Logs

### Problem: "Why did this fail?"

**Look at logs:**
```bash
tail -100 logs/error.log
```

**Find:**
- When it failed (timestamp)
- What was being processed (context)
- What the error was (message + traceback)

### Problem: "How long does this take?"

**Use log_performance:**
```python
@log_performance(logger)
def slow_operation():
    # Your code
    pass
```

**Check logs:**
```
INFO: slow_operation took 5.23s
```

### Problem: "Did this run?"

**Add logs:**
```python
logger.info("Starting critical operation")
# Code
logger.info("Critical operation completed")
```

**Check logs:**
```bash
grep "critical operation" logs/app.log
```

---

## 📚 Real Example

### Before (No Logging):

```python
def search_web(query):
    results = api.search(query)
    parsed = parse_results(results)
    return parsed
```

**When it breaks:** 🤷 No idea why

---

### After (With Logging):

```python
from logging_config import get_logger, LogContext

logger = get_logger(__name__)

def search_web(query: str):
    logger.info(f"Web search requested: {query}")
    
    with LogContext("api_call", logger):
        try:
            results = api.search(query)
            logger.debug(f"API returned {len(results)} results")
            
        except APIError as e:
            logger.error(f"API call failed for query: {query}", exc_info=True)
            raise
    
    with LogContext("parsing", logger):
        parsed = parse_results(results)
        logger.info(f"Parsed {len(parsed)} valid results")
    
    return parsed
```

**When it breaks:** Check logs:
```
INFO: Web search requested: python tutorial
INFO: Starting api_call
ERROR: API call failed for query: python tutorial
Traceback (most recent call last):
  ...
  APIError: Rate limit exceeded
```

**Now you know:** Rate limit hit during API call!

---

## 🔄 Log Rotation (Already Configured)

**Your logs automatically rotate:**
- Max size: 10 MB per file
- Keeps: 5 backup files
- Oldest logs deleted automatically

**Files:**
```
logs/
  app.log           # Current
  app.log.1         # Yesterday
  app.log.2         # 2 days ago
  app.log.3         # 3 days ago
  app.log.4         # 4 days ago
  app.log.5         # 5 days ago
```

**You don't need to do anything!**

---

## ✅ Quick Checklist

For your project:

- [ ] Import logging_config in main.py
- [ ] Call setup_logging() at startup
- [ ] Use get_logger(__name__) in each file
- [ ] Log when functions start
- [ ] Log errors with exc_info=True
- [ ] Check logs/ directory is in .gitignore
- [ ] Use LogContext for timed operations

---

## 🎯 Quick Reference

### Import and Setup:
```python
from logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)
```

### Basic Logging:
```python
logger.debug("Detailed info")
logger.info("General info")
logger.warning("Something unexpected")
logger.error("Something failed", exc_info=True)
logger.critical("System failing", exc_info=True)
```

### Timed Operations:
```python
with LogContext("operation_name", logger):
    do_something()
```

### Performance Decorator:
```python
@log_performance(logger)
def function():
    pass
```

---

## 💡 Pro Tips

1. **Log business events** - "Order placed", "User registered"
2. **Don't log passwords** - Even in DEBUG mode
3. **Use meaningful messages** - "User login failed: invalid password"
4. **Include IDs** - user_id, order_id for tracking
5. **Check logs regularly** - Catch issues early
6. **Use grep to search** - `grep "ERROR" logs/app.log`

---

## 🆘 Common Issues

### Issue: "No logs appearing"

**Check:**
```python
# Did you call setup_logging()?
setup_logging()

# Did you create logger?
logger = get_logger(__name__)

# Are you using logger, not print?
logger.info("message")  # Not print("message")
```

---

### Issue: "Too many logs"

**Solution:**
```bash
# Increase log level
# In .env:
LOG_LEVEL=INFO  # Instead of DEBUG
```

---

### Issue: "Can't find error"

**Solution:**
```bash
# Check error log
tail -100 logs/error.log

# Search by date
grep "2024-01-15" logs/app.log

# Search by keyword
grep -i "user" logs/app.log
```

---

**Logging = Your debugging superpower!** 🚀

---

*For full logging examples, see src/logging_config.py*
*For git workflow, see [GIT_WORKFLOW.md](GIT_WORKFLOW.md)*
