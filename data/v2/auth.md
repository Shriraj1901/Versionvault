# Authentication (v2)

API v2 replaces query-param API keys with Bearer tokens.

Include your token in the Authorization header:

    Authorization: Bearer YOUR_TOKEN

Tokens expire after 24 hours and must be refreshed using the
POST /v2/auth/refresh endpoint. Query-param API keys from v1 are
no longer accepted in v2.