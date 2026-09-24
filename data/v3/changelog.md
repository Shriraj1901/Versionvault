# Changelog (v3)

- Switched authentication from Bearer tokens to full OAuth 2.0.
- Added `role` field and filter to the /users endpoint.
- Deprecated v2 Bearer-only auth (still accepted temporarily).
- Improved pagination defaults: limit now defaults to 50 (was 20 in v2).