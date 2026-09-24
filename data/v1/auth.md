# Authentication (v1)

API v1 uses simple API key authentication.

Include your API key as a query parameter on every request:

    GET /v1/users?api_key=YOUR_KEY

There is no support for OAuth or token expiry in v1. Keys do not expire
and must be manually rotated by contacting support.