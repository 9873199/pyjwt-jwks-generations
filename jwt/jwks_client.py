from __future__ import annotations

import http.client
import json
import math
import threading
import time
import urllib.request
from collections import OrderedDict
from ssl import SSLContext
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class _KeyGeneration:
    """An immutable view of one successfully accepted JWKS response.

    A generation is published atomically: every key handed out while it is
    current comes from the same :class:`PyJWKSet`, so a query can never
    observe a mix of two endpoint responses. Retiring a ``kid`` or rotating
    the key behind a ``kid`` therefore means publishing a new generation;
    every per-key lookup cached against the old generation dies with it.
    """

    def __init__(
        self,
        generation_id: int,
        jwk_set: PyJWKSet,
        published_at: float,
        source: Any = None,
    ) -> None:
        self.generation_id = generation_id
        self.jwk_set = jwk_set
        self.published_at = published_at
        # The raw value this generation was accepted from, kept by identity
        # so `get_jwk_set()` can reuse the single parse performed by
        # `fetch_data()` instead of parsing the same payload twice.
        self.source = source
        # Per-kid memoization scoped to this generation only. Bounded like
        # the previous LRU cache, but never shared across generations, so it
        # cannot keep a retired or rotated key alive. Only ever touched while
        # the client's lock is held.
        self._key_index: OrderedDict[str, PyJWK] = OrderedDict()

    def lookup_kid(self, kid: str, max_cached_keys: int | None) -> PyJWK | None:
        """Return the key for *kid* in this generation, or ``None``.

        When *max_cached_keys* is set, positive lookups are memoized in a
        bounded LRU index owned by this generation.
        """
        if max_cached_keys is not None:
            cached = self._key_index.get(kid)
            if cached is not None:
                self._key_index.move_to_end(kid)
                return cached

        signing_key = PyJWKClient.match_kid(
            PyJWKClient._get_signing_keys_from_jwk_set(self.jwk_set), kid
        )

        if signing_key is not None and max_cached_keys is not None:
            self._key_index[kid] = signing_key
            while len(self._key_index) > max_cached_keys:
                self._key_index.popitem(last=False)

        return signing_key


class PyJWKClient:
    def __init__(
        self,
        uri: str,
        cache_keys: bool = False,
        max_cached_keys: int = 16,
        cache_jwk_set: bool = True,
        lifespan: float = 300,
        headers: dict[str, Any] | None = None,
        timeout: float = 30,
        ssl_context: SSLContext | None = None,
        cooldown_duration: float = 30,
    ):
        """A client for retrieving signing keys from a JWKS endpoint.

        ``PyJWKClient`` maintains a single, generation-based cache of the
        endpoint's keys instead of two independent caches:

        **Key generations** (enabled by default via ``cache_jwk_set``):
        Every successfully accepted JWKS response is published as a new
        immutable :class:`~jwt.api_jwk.PyJWKSet` snapshot ("generation").
        All queries served while a generation is current see exactly the
        same key objects:

        - A ``kid`` removed from the endpoint disappears as soon as the next
          JWKS is accepted; lookups cached against the old generation cannot
          keep the retired key alive.
        - A ``kid`` whose public key changed resolves to the new key object
          after the refresh; the old object is never returned again.
        - A failed download or a response that is not a valid JWKS publishes
          nothing, so the previous generation stays available until its TTL
          expires. Once expired, a failed refresh is raised to the caller
          rather than serving stale keys indefinitely.

        Controlled by:

        - ``cache_jwk_set``: Set to ``True`` (the default) to keep the
          latest accepted generation and only fetch when it is missing or
          expired. Set to ``False`` to fetch from the network on every call.
        - ``lifespan``: Time in seconds before a generation expires.
          Defaults to ``300`` (5 minutes). Must be greater than 0.

        Unknown key IDs can trigger one forced refresh after the cooldown
        period configured by ``cooldown_duration``. Every successful fetch
        starts this cooldown, including the initial fetch and cache-expiry
        fetches. A newly rotated key may therefore wait for the cooldown
        period before it is fetched; set ``cooldown_duration`` to ``0`` to
        disable this behavior. The cooldown is bypassed when
        ``cache_jwk_set`` is ``False``.

        **Per-key memoization** (disabled by default):
        When ``cache_keys`` is ``True``, individual ``kid`` lookups are
        memoized *within the current generation* in a bounded LRU index.
        Unlike a standalone key cache, the index is discarded wholesale
        whenever a new generation is published, so its entries can never
        disagree with the JWKS they came from. Because
        ``cache_jwk_set=False`` makes every call accept a fresh generation,
        this memo then never spans two calls. Controlled by:

        - ``cache_keys``: Set to ``True`` to enable per-key memoization.
          Defaults to ``False``.
        - ``max_cached_keys``: Maximum number of lookups memoized per
          generation. Defaults to ``16``.

        :param uri: The URL of the JWKS endpoint.
        :type uri: str
        :param cache_keys: Enable per-key memoization within a generation.
        :type cache_keys: bool
        :param max_cached_keys: Max memoized lookups per generation.
        :type max_cached_keys: int
        :param cache_jwk_set: Keep the latest accepted JWKS generation.
        :type cache_jwk_set: bool
        :param lifespan: TTL in seconds for a cached generation.
        :type lifespan: float
        :param headers: Optional HTTP headers to include in requests.
        :type headers: dict or None
        :param timeout: HTTP request timeout in seconds.
        :type timeout: float
        :param ssl_context: Optional SSL context for the request.
        :type ssl_context: ssl.SSLContext or None
        :param cooldown_duration: Minimum time in seconds between forced
            refreshes after an unknown key ID. Defaults to ``30``.
        :type cooldown_duration: float
        """
        if headers is None:
            headers = {}
        # urllib's default OpenerDirector also handles file://, ftp://, and
        # data: URIs. Reject anything that isn't http(s) eagerly so a caller
        # passing an attacker-influenced URL (e.g. taken from a `jku` token
        # header) can't read local files or reach other unintended schemes.
        scheme = urlparse(uri).scheme.lower()
        if scheme not in ("http", "https"):
            raise PyJWKClientError(
                f"Invalid JWKS URI scheme {scheme!r}: only 'http' and 'https' "
                f"are supported."
            )
        self.uri = uri
        self.jwk_set_cache: JWKSetCache | None = None
        self.headers = headers
        self.timeout = timeout
        self.ssl_context = ssl_context
        if cooldown_duration < 0:
            raise PyJWKClientError(
                "Cooldown duration must be greater than or equal to 0, "
                f'the input is "{cooldown_duration}"'
            )
        if not math.isfinite(cooldown_duration):
            raise PyJWKClientError(
                f'Cooldown duration must be finite, the input is "{cooldown_duration}"'
            )
        self.cooldown_duration = cooldown_duration
        self.cache_keys = cache_keys
        self.max_cached_keys = max_cached_keys
        self._last_successful_fetch: float | None = None
        self._client_lock = threading.RLock()
        self._generation: _KeyGeneration | None = None
        self._next_generation_id = 0

        if cache_jwk_set:
            # Init jwt set cache with default or given lifespan.
            # Default lifespan is 300 seconds (5 minutes).
            if lifespan <= 0:
                raise PyJWKClientError(
                    f'Lifespan must be greater than 0, the input is "{lifespan}"'
                )
            self.jwk_set_cache = JWKSetCache(lifespan)
        else:
            self.jwk_set_cache = None

    def _publish_generation(
        self, jwk_set: PyJWKSet, source: Any = None
    ) -> _KeyGeneration:
        """Atomically make *jwk_set* the current key generation.

        The previous generation (and everything memoized against it) becomes
        unreachable through the client once the replacement is assigned.
        *source* is the raw payload (if any) that *jwk_set* was parsed from,
        kept by identity so a later call can tell an overridden
        ``fetch_data()`` transformed the response.
        """
        with self._client_lock:
            self._next_generation_id += 1
            generation = _KeyGeneration(
                self._next_generation_id, jwk_set, time.monotonic(), source
            )
            self._generation = generation
            if self.jwk_set_cache is not None:
                self.jwk_set_cache.put(jwk_set)
            self._last_successful_fetch = generation.published_at
            return generation

    def _is_generation_expired(self, generation: _KeyGeneration) -> bool:
        if self.jwk_set_cache is None:
            # With the JWKS cache disabled every call fetches fresh data;
            # a retained generation is never served, so it never goes stale.
            return True
        return time.monotonic() > generation.published_at + self.jwk_set_cache.lifespan

    def fetch_data(self) -> Any:
        """Fetch the JWK Set from the JWKS endpoint.

        Makes an HTTP request to the configured ``uri`` and returns the
        parsed JSON response. On success the response is accepted as a new
        key generation and stored in the JWK Set cache. A network failure
        or a payload that is not a JSON object raises before any new state
        is published, leaving the previously accepted generation untouched.

        :returns: The parsed JWK Set as a dictionary.
        :raises PyJWKClientConnectionError: If the HTTP request fails.
        :raises PyJWKClientError: If the endpoint does not return a JSON
            object.
        :raises PyJWKSetError: If the JWK Set cache is enabled and the
            response contains no usable keys.
        """
        try:
            r = urllib.request.Request(url=self.uri, headers=self.headers)
            handlers: list[Any] = [_NoRedirectHandler()]
            if self.ssl_context is not None:
                handlers.append(urllib.request.HTTPSHandler(context=self.ssl_context))
            opener = urllib.request.build_opener(*handlers)
            with opener.open(r, timeout=self.timeout) as response:
                raw_data = json.load(response)
        except (URLError, TimeoutError, http.client.HTTPException) as e:
            if isinstance(e, HTTPError):
                e.close()
            raise PyJWKClientConnectionError(
                f'Fail to fetch data from the url, err: "{e}"'
            ) from e

        # Validate the payload before it reaches the cache, so an endpoint
        # returning something that isn't a JSON object is reported as such
        # rather than as a malformed JWK Set.
        payload = self._as_jwk_set_payload(raw_data)

        # Accept the response as a new generation only after the fetch and
        # payload validation succeeded. Writing in a `finally` block with
        # `jwk_set=None` on error would clear any previously-cached JWKS,
        # turning a transient outage into a cache wipe that breaks
        # legitimate auth. The payload is parsed once here and the same
        # `PyJWKSet` is what later lookups serve.
        jwk_set = PyJWKSet.from_dict(payload)
        self._publish_generation(jwk_set, source=payload)
        return payload

    def get_jwk_set(self, refresh: bool = False) -> PyJWKSet:
        """Return the JWK Set, using the current generation when available.

        :param refresh: Force a fresh fetch from the endpoint, bypassing
            the cache.
        :type refresh: bool
        :returns: The JWK Set.
        :rtype: PyJWKSet
        :raises PyJWKClientError: If the endpoint does not return a JSON
            object.
        """
        with self._client_lock:
            generation = self._generation

            if (
                not refresh
                and generation is not None
                and not self._is_generation_expired(generation)
            ):
                return generation.jwk_set

            if not refresh and self.jwk_set_cache is not None:
                # The cache may have been pre-populated by the caller; adopt
                # its value as the current generation without a fetch and
                # without resetting the TTL the cache already assigned it.
                cached = self.jwk_set_cache.get()
                if cached is not None:
                    with_timestamp = self.jwk_set_cache.jwk_set_with_timestamp
                    assert with_timestamp is not None
                    cached_at = with_timestamp.get_timestamp()
                    self._next_generation_id += 1
                    generation = _KeyGeneration(
                        self._next_generation_id, cached, cached_at
                    )
                    self._generation = generation
                    return cached

            # Either a forced refresh, an expired generation, or no cached
            # state. `fetch_data()` validates the response and publishes the
            # new generation itself; a failure raises before any state
            # changes, so the previous generation stays available.
            data = self.fetch_data()
            generation = self._generation

            # A subclass may override `fetch_data()` without publishing a
            # generation, or filter/transform its result. Publish the value
            # actually returned so the current generation matches this call
            # rather than the pre-transform one (or the missing one).
            if generation is None or generation.source is not data:
                if isinstance(data, PyJWKSet):
                    jwk_set = data
                else:
                    jwk_set = PyJWKSet.from_dict(self._as_jwk_set_payload(data))
                if generation is None or jwk_set is not generation.jwk_set:
                    generation = self._publish_generation(jwk_set)

            return generation.jwk_set

    @staticmethod
    def _as_jwk_set_payload(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise PyJWKClientError("The JWKS endpoint did not return a JSON object")

        return data

    def get_signing_keys(self, refresh: bool = False) -> list[PyJWK]:
        """Return all signing keys from the current JWKS generation.

        Filters the JWK Set to keys whose ``use`` is ``"sig"`` (or
        unspecified) and that have a ``kid``.

        :param refresh: Force a fresh fetch from the endpoint, bypassing
            the cache.
        :type refresh: bool
        :returns: A list of signing keys.
        :rtype: list[PyJWK]
        :raises PyJWKClientError: If no signing keys are found.
        """
        jwk_set = self.get_jwk_set(refresh)
        return self._get_signing_keys_from_jwk_set(jwk_set)

    @staticmethod
    def _get_signing_keys_from_jwk_set(jwk_set: PyJWKSet) -> list[PyJWK]:
        signing_keys = [
            jwk_set_key
            for jwk_set_key in jwk_set.keys
            if jwk_set_key.public_key_use in ["sig", None] and jwk_set_key.key_id
        ]

        if not signing_keys:
            raise PyJWKClientError("The JWKS endpoint did not contain any signing keys")

        return signing_keys

    def get_signing_key(self, kid: str) -> PyJWK:
        """Return the signing key matching the given ``kid``.

        The lookup is resolved against one immutable key generation. If no
        match is found in the current generation, the JWKS is refreshed
        from the endpoint and the lookup is retried once against the newly
        published generation when the refresh cooldown permits it.

        :param kid: The key ID to look up.
        :type kid: str
        :returns: The matching signing key.
        :rtype: PyJWK
        :raises PyJWKClientError: If no matching key is found after
            refreshing.
        """
        with self._client_lock:
            generation = self._require_generation()
            signing_key = generation.lookup_kid(
                kid, self.max_cached_keys if self.cache_keys else None
            )

            if not signing_key:
                cooling_down = (
                    self.jwk_set_cache is not None
                    and self._last_successful_fetch is not None
                    and time.monotonic() - self._last_successful_fetch
                    < self.cooldown_duration
                )
                if not cooling_down:
                    self.get_jwk_set(refresh=True)
                    # `_publish_generation()` already recorded the fetch
                    # time. Retry strictly against the generation just
                    # published; the old generation's memoized answer is
                    # never mixed in.
                    refreshed_generation = self._generation
                    assert refreshed_generation is not None
                    signing_key = refreshed_generation.lookup_kid(
                        kid, self.max_cached_keys if self.cache_keys else None
                    )

                if not signing_key:
                    raise PyJWKClientError(
                        f'Unable to find a signing key that matches: "{kid}"'
                    )

            return signing_key

    def _require_generation(self) -> _KeyGeneration:
        """Return the current generation, accepting a JWKS if none exists."""
        if self._generation is None or self._is_generation_expired(self._generation):
            self.get_jwk_set()
        generation = self._generation
        assert generation is not None
        return generation

    def get_signing_key_from_jwt(self, token: str | bytes) -> PyJWK:
        """Return the signing key for a JWT by reading its ``kid`` header.

        Extracts the ``kid`` from the token's unverified header and
        delegates to :meth:`get_signing_key`.

        :param token: The encoded JWT.
        :type token: str or bytes
        :returns: The matching signing key.
        :rtype: PyJWK
        """
        unverified = decode_token(token, options={"verify_signature": False})
        header = unverified["header"]
        return self.get_signing_key(header.get("kid"))

    @staticmethod
    def match_kid(signing_keys: list[PyJWK], kid: str) -> PyJWK | None:
        """Find a key in *signing_keys* that matches *kid*.

        :param signing_keys: The list of keys to search.
        :type signing_keys: list[PyJWK]
        :param kid: The key ID to match.
        :type kid: str
        :returns: The matching key, or ``None`` if no match.
        :rtype: PyJWK or None
        """
        signing_key = None

        for key in signing_keys:
            if key.key_id == kid:
                signing_key = key
                break

        return signing_key
