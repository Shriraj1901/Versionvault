# Authentication (v3)

API v3 uses OAuth 2.0 with short-lived access tokens and refresh tokens.

Obtain a token via POST /v3/oauth/token using your client_id and
client_secret. Access tokens expire after 1 hour. Use the returned
refresh_token to obtain a new access token without re-authenticating.

Bearer-token-only auth from v2 is deprecated but still accepted until
the next major version.