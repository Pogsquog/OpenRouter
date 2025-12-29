Security Best Practices
=======================

The OpenRouter Python library implements several security best practices to protect your API keys and sensitive data.

API Key Handling
----------------

1. **Hashed Logging**: API keys are never logged directly. Instead, a hash of the API key is used for logging purposes to identify which key was used without exposing its value.

2. **Environment Variables**: The library supports loading API keys from environment variables (``OPENROUTER_API_KEY``) to avoid hardcoding them in source code.

3. **Input Validation**: All API keys are validated using regex patterns to ensure they follow the expected format before being used.

4. **Secure Transmission**: API keys are transmitted securely over HTTPS using the standard Authorization header.

Request Logging
---------------

By default, request logging is kept minimal to prevent sensitive data exposure. Detailed request logging can be enabled with the ``enable_request_logging`` parameter, but this should only be used in secure environments for debugging purposes.

.. code-block:: python

    # Enable detailed logging (use with caution in production)
    client = AsyncOpenRouter(
        api_key="your-api-key",
        enable_request_logging=True  # Only enable for debugging
    )

Additional Security Measures
----------------------------

* All HTTP requests use HTTPS by default
* Request timeouts prevent hanging connections
* Rate limiting is handled automatically
* Error messages do not expose sensitive internal information
* The library follows secure coding practices to prevent common vulnerabilities