"""Generation-based key view tests for :class:`PyJWKClient`.

These tests pin the cache semantics for key retirement and rotation:

* every successfully accepted JWKS becomes a new immutable generation,
* per-kid memoization is scoped to a generation and dies with it,
* failed or invalid refreshes publish nothing and stop being served once
  the previous generation's TTL expires,
* the unknown-kid refresh cooldown still protects the endpoint, and
* concurrent refresh/lookup interleaving can never mix two generations.

Timing is driven by a mocked monotonic clock plus ``threading`` barriers
and events -- nothing here sleeps to wait out a timer. Tokens are really
signed with locally generated EC keys so the full "select key, then verify
signature" call chain is exercised.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from typing import Any
from unittest import mock
from urllib.error import URLError

import pytest

import jwt
from jwt import PyJWKClient
from jwt.algorithms import has_crypto
from jwt.exceptions import (
    InvalidSignatureError,
    PyJWKClientConnectionError,
    PyJWKClientError,
)

from .utils import crypto_required

if has_crypto:
    from cryptography.hazmat.primitives.asymmetric import ec

    from jwt.algorithms import ECAlgorithm

URL = "https://example.test/.well-known/jwks.json"


def _generate_ec_private_key() -> Any:
    return ec.generate_private_key(ec.SECP256R1())


def _public_jwk(private_key: Any, kid: str) -> dict[str, Any]:
    jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk["kid"] = kid
    jwk["use"] = "sig"
    jwk["alg"] = "ES256"
    return jwk


def _jwks(*keys: dict[str, Any]) -> dict[str, Any]:
    return {"keys": list(keys)}


def _signed_token(private_key: Any, kid: str) -> str:
    return jwt.encode(
        {"sub": f"token-signed-by-{kid}"},
        private_key,
        algorithm="ES256",
        headers={"kid": kid},
    )


@contextlib.contextmanager
def mocked_endpoint(
    payloads: list[Any], on_fetch: Any = None
) -> Iterator[tuple[mock.Mock, list[Any]]]:
    """Serve *payloads* in order for successive ``opener.open`` calls.

    A payload that is an :class:`Exception` is raised instead of returned,
    modeling a failed download. Any fetch beyond the listed payloads fails
    loudly so tests can assert exact request counts. *on_fetch*, when given,
    is called with the zero-based fetch index before each response is built,
    letting a test block a request inside a synchronization barrier.
    """
    with mock.patch("urllib.request.build_opener") as build_opener_mock:
        opener = mock.Mock()
        build_opener_mock.return_value = opener
        requests: list[Any] = []

        def open_side_effect(*args: Any, **kwargs: Any) -> mock.Mock:
            index = len(requests)
            requests.append(args[0] if args else None)
            if on_fetch is not None:
                on_fetch(index)
            if index >= len(payloads):
                raise AssertionError("unexpected extra JWKS fetch")
            payload = payloads[index]
            if isinstance(payload, Exception):
                raise payload
            response = mock.Mock()
            response.__enter__ = mock.Mock(return_value=response)
            response.__exit__ = mock.Mock()
            response.read.return_value = json.dumps(payload)
            return response

        opener.open.side_effect = open_side_effect
        yield opener.open, requests


@contextlib.contextmanager
def controllable_clock(start: float = 0.0) -> Iterator[list[float]]:
    clock = [start]
    with mock.patch("jwt.jwks_client.time.monotonic", side_effect=lambda: clock[0]):
        yield clock


def current_generation(client: PyJWKClient) -> Any:
    """Return the client's current generation, asserting one was published."""
    generation = client._generation
    assert generation is not None
    return generation


@crypto_required
class TestKeyGenerationCache:
    @pytest.mark.parametrize("cache_keys", [True, False])
    def test_retired_kid_stops_being_selected_after_refresh(
        self, cache_keys: bool
    ) -> None:
        # First JWKS has A and B; the client memoizes A. The endpoint then
        # retires A and only serves B. A must not be selectable anymore,
        # while B keeps verifying its own tokens through get_signing_key().
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwk_a = _public_jwk(key_a, "A")
        jwk_b = _public_jwk(key_b, "B")
        token_a = _signed_token(key_a, "A")
        token_b = _signed_token(key_b, "B")

        client = PyJWKClient(URL, cache_keys=cache_keys, lifespan=100)
        with mocked_endpoint([_jwks(jwk_a, jwk_b), _jwks(jwk_b)]) as (
            open_mock,
            _requests,
        ):
            selected_a = client.get_signing_key("A")
            selected_b = client.get_signing_key("B")
            assert current_generation(client).generation_id == 1
            jwt.decode(token_a, selected_a.key, algorithms=["ES256"])
            jwt.decode(token_b, selected_b.key, algorithms=["ES256"])

            client.get_jwk_set(refresh=True)
            assert current_generation(client).generation_id == 2

            # The retired key is gone from every entry point, including the
            # JWT header driven one.
            with pytest.raises(PyJWKClientError, match='matches: "A"'):
                client.get_signing_key("A")
            with pytest.raises(PyJWKClientError, match='matches: "A"'):
                client.get_signing_key_from_jwt(token_a)
            assert [key.key_id for key in client.get_signing_keys()] == ["B"]

            # B remains in the list and still verifies its token through
            # both the kid-based and the JWT-header-driven entry points.
            selected_b_again = client.get_signing_key("B")
            jwt.decode(token_b, selected_b_again.key, algorithms=["ES256"])
            b_from_jwt = client.get_signing_key_from_jwt(token_b)
            jwt.decode(token_b, b_from_jwt.key, algorithms=["ES256"])

            # The old key object still verifies the old token on its own --
            # retirement removes it from selection, it does not destroy the
            # immutable snapshot an earlier query belongs to.
            jwt.decode(token_a, selected_a.key, algorithms=["ES256"])
            assert open_mock.call_count == 2

    @pytest.mark.parametrize("cache_keys", [True, False])
    def test_same_kid_with_new_public_key_replaces_the_object(
        self, cache_keys: bool
    ) -> None:
        # A keeps its kid but is rotated to a new key pair. The post-refresh
        # view must hand out the new object; a token signed by the old key
        # has to fail signature verification against it.
        old_a = _generate_ec_private_key()
        new_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        old_jwk_a = _public_jwk(old_a, "A")
        new_jwk_a = _public_jwk(new_a, "A")
        jwk_b = _public_jwk(key_b, "B")
        old_token = _signed_token(old_a, "A")
        new_token = _signed_token(new_a, "A")

        client = PyJWKClient(URL, cache_keys=cache_keys, lifespan=100)
        with mocked_endpoint([_jwks(old_jwk_a, jwk_b), _jwks(new_jwk_a, jwk_b)]):
            first = client.get_signing_key("A")
            jwt.decode(old_token, first.key, algorithms=["ES256"])
            if cache_keys:
                # Memoization returns the same object while the generation
                # has not changed.
                assert client.get_signing_key("A") is first

            client.get_jwk_set(refresh=True)
            second = client.get_signing_key("A")

            assert second is not first
            jwt.decode(new_token, second.key, algorithms=["ES256"])
            with pytest.raises(InvalidSignatureError):
                jwt.decode(old_token, second.key, algorithms=["ES256"])

    def test_failed_refresh_keeps_old_view_until_ttl_then_refuses_stale(
        self,
    ) -> None:
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwks_ab = _jwks(_public_jwk(key_a, "A"), _public_jwk(key_b, "B"))

        client = PyJWKClient(URL, lifespan=100)
        with mocked_endpoint([jwks_ab, URLError("boom"), URLError("still down")]) as (
            open_mock,
            _requests,
        ):
            with controllable_clock(0.0) as clock:
                client.get_jwk_set()
                assert current_generation(client).generation_id == 1

                # Well inside the TTL, a failed refresh publishes nothing and
                # the previous generation stays usable without another fetch.
                with pytest.raises(PyJWKClientConnectionError):
                    client.get_jwk_set(refresh=True)
                assert current_generation(client).generation_id == 1
                assert client.get_signing_key("A").key_id == "A"

                # Past the TTL there is no infinite fallback: a failed fetch
                # is raised instead of serving an expired generation.
                clock[0] = 150.0
                with pytest.raises(PyJWKClientConnectionError):
                    client.get_signing_key("A")
                assert open_mock.call_count == 3

    def test_invalid_payload_does_not_publish_a_generation(self) -> None:
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwks_ab = _jwks(_public_jwk(key_a, "A"), _public_jwk(key_b, "B"))
        jwks_b = _jwks(_public_jwk(key_b, "B"))

        client = PyJWKClient(URL, lifespan=100)
        with mocked_endpoint([jwks_ab, ["not", "a", "json", "object"], jwks_b]):
            with controllable_clock(0.0):
                client.get_jwk_set()
                first_generation = current_generation(client)
                assert first_generation.generation_id == 1

                with pytest.raises(PyJWKClientError, match="did not return a JSON"):
                    client.get_jwk_set(refresh=True)
                # Nothing was published: same generation object, A still
                # selectable from the old view.
                assert client._generation is first_generation
                assert client.get_signing_key("A").key_id == "A"

                client.get_jwk_set(refresh=True)
                assert current_generation(client).generation_id == 2
                with pytest.raises(PyJWKClientError):
                    client.get_signing_key("A")

    def test_unknown_kid_cooldown_still_protects_the_endpoint(self) -> None:
        key_b = _generate_ec_private_key()
        jwks_b = _jwks(_public_jwk(key_b, "B"))

        client = PyJWKClient(URL, cooldown_duration=30, lifespan=100)
        with mocked_endpoint([jwks_b, jwks_b]) as (open_mock, _requests):
            with controllable_clock(0.0) as clock:
                # Initial fetch plus a missed lookup; cooldown starts now.
                with pytest.raises(PyJWKClientError, match='matches: "random"'):
                    client.get_signing_key("random")
                assert open_mock.call_count == 1

                # Still cooling down: no forced refresh may hit the endpoint.
                clock[0] = 10.0
                with pytest.raises(PyJWKClientError, match='matches: "random"'):
                    client.get_signing_key("random")
                assert open_mock.call_count == 1

                # After the cooldown a single forced refresh is allowed, and
                # it still misses.
                clock[0] = 40.0
                with pytest.raises(PyJWKClientError, match='matches: "random"'):
                    client.get_signing_key("random")
                assert open_mock.call_count == 2

    def test_two_threads_racing_a_refresh_share_one_generation(self) -> None:
        # Gen 1 (A, B) expires; the endpoint now serves only B. Two threads
        # cross a barrier and race the refresh. Locking must serialize them
        # into exactly one fetch and both queries must resolve against the
        # same new generation -- never a mix of old and new key objects.
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwks_ab = _jwks(_public_jwk(key_a, "A"), _public_jwk(key_b, "B"))
        jwks_b = _jwks(_public_jwk(key_b, "B"))

        client = PyJWKClient(URL, cache_keys=True, lifespan=100)
        refresh_started = threading.Event()
        release_refresh = threading.Event()
        start_barrier = threading.Barrier(2)
        results: list[Any] = []
        errors: list[Exception] = []

        def on_fetch(index: int) -> None:
            # Fetch 0 is the initial one; block the racing refresh (fetch 1)
            # while the second thread queues up behind the client lock.
            if index == 1:
                refresh_started.set()
                assert release_refresh.wait(timeout=5)

        with mocked_endpoint([jwks_ab, jwks_b], on_fetch=on_fetch) as (
            open_mock,
            _requests,
        ):
            with controllable_clock(0.0) as clock:
                client.get_jwk_set()
                old_generation = current_generation(client)
                assert old_generation.generation_id == 1
                # Warm the old generation's per-kid index.
                client.get_signing_key("A")
                assert "A" in old_generation._key_index

                clock[0] = 150.0  # gen 1 is now expired

                def lookup() -> None:
                    try:
                        start_barrier.wait(timeout=5)
                        results.append(client.get_jwk_set())
                    except Exception as error:  # noqa: BLE001 - surfaced below
                        errors.append(error)

                first = threading.Thread(target=lookup)
                second = threading.Thread(target=lookup)
                first.start()
                second.start()
                assert refresh_started.wait(timeout=5)
                # The refresh (the second fetch) is in flight and blocked;
                # both threads must still be alive and no third fetch may
                # have been started by the queued second thread.
                assert first.is_alive()
                assert second.is_alive()
                assert open_mock.call_count == 2
                release_refresh.set()
                first.join(timeout=5)
                second.join(timeout=5)

        assert errors == []
        assert not first.is_alive()
        assert not second.is_alive()
        # Initial fetch plus exactly one racing refresh: the queued thread
        # reused the freshly published generation instead of fetching again.
        assert open_mock.call_count == 2
        new_generation = current_generation(client)
        assert new_generation.generation_id == 2
        assert results[0] is results[1] is new_generation.jwk_set
        # The new generation's view never inherited the old memoized A entry.
        assert new_generation._key_index == {}
        with pytest.raises(PyJWKClientError):
            client.get_signing_key("A")
        # The retired generation remains a coherent frozen snapshot.
        assert old_generation.lookup_kid("A", None) is not None

    def test_concurrent_signing_key_lookups_belong_to_one_generation(self) -> None:
        # Same race as above but through get_signing_key(): B exists in both
        # generations, so each racing lookup must return the *new* B object
        # from one shared generation, never a key that belongs to the
        # expired one (a half-new, half-old view).
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwks_ab = _jwks(_public_jwk(key_a, "A"), _public_jwk(key_b, "B"))
        jwks_b = _jwks(_public_jwk(key_b, "B"))

        client = PyJWKClient(URL, cache_keys=True, lifespan=100)
        refresh_started = threading.Event()
        release_refresh = threading.Event()
        start_barrier = threading.Barrier(2)
        keys: list[Any] = []
        errors: list[Exception] = []

        def on_fetch(index: int) -> None:
            if index == 1:
                refresh_started.set()
                assert release_refresh.wait(timeout=5)

        with mocked_endpoint([jwks_ab, jwks_b], on_fetch=on_fetch) as (
            open_mock,
            _requests,
        ):
            with controllable_clock(0.0) as clock:
                client.get_jwk_set()
                old_b = client.get_signing_key("B")
                clock[0] = 150.0  # gen 1 expired

                def lookup() -> None:
                    try:
                        start_barrier.wait(timeout=5)
                        keys.append(client.get_signing_key("B"))
                    except Exception as error:  # noqa: BLE001 - surfaced below
                        errors.append(error)

                first = threading.Thread(target=lookup)
                second = threading.Thread(target=lookup)
                first.start()
                second.start()
                assert refresh_started.wait(timeout=5)
                release_refresh.set()
                first.join(timeout=5)
                second.join(timeout=5)

        assert errors == []
        assert open_mock.call_count == 2
        new_generation = current_generation(client)
        assert new_generation.generation_id == 2
        # Both answers are the same new-generation object, never the old one.
        assert len(keys) == 2
        assert keys[0] is keys[1] is new_generation.lookup_kid("B", None)
        assert keys[0] is not old_b

    def test_direct_fetch_data_publishes_and_retires_generations(self) -> None:
        # The fetch_data() entry point must maintain the same unified view
        # as get_signing_key(): each successful call is a new generation and
        # a failure leaves the current one untouched.
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        jwk_a = _public_jwk(key_a, "A")
        jwk_b = _public_jwk(key_b, "B")
        jwks_ab = _jwks(jwk_a, jwk_b)
        jwks_b = _jwks(jwk_b)

        client = PyJWKClient(URL, cache_keys=True, lifespan=100)
        with mocked_endpoint([jwks_ab, jwks_b, URLError("boom")]) as (
            open_mock,
            _requests,
        ):
            assert client.fetch_data() == jwks_ab
            assert current_generation(client).generation_id == 1
            # Memoized lookup is served without another network call.
            assert client.get_signing_key("A").key_id == "A"
            assert open_mock.call_count == 1

            assert client.fetch_data() == jwks_b
            assert current_generation(client).generation_id == 2
            with pytest.raises(PyJWKClientError):
                client.get_signing_key("A")
            assert client.get_signing_key("B").key_id == "B"

            with pytest.raises(PyJWKClientConnectionError):
                client.fetch_data()
            assert current_generation(client).generation_id == 2
            assert client.get_signing_key("B").key_id == "B"

    def test_per_key_memoization_is_bounded_per_generation(self) -> None:
        key_a = _generate_ec_private_key()
        key_b = _generate_ec_private_key()
        key_c = _generate_ec_private_key()
        jwks_abc = _jwks(
            _public_jwk(key_a, "A"),
            _public_jwk(key_b, "B"),
            _public_jwk(key_c, "C"),
        )
        jwks_bc = _jwks(_public_jwk(key_b, "B"), _public_jwk(key_c, "C"))

        client = PyJWKClient(URL, cache_keys=True, max_cached_keys=2, lifespan=100)
        with mocked_endpoint([jwks_abc, jwks_bc]):
            client.get_signing_key("A")
            b_key = client.get_signing_key("B")
            client.get_signing_key("C")  # evicts A from the bounded index
            generation = current_generation(client)
            assert list(generation._key_index) == ["B", "C"]
            assert client.get_signing_key("B") is b_key

            client.get_jwk_set(refresh=True)
            # A fresh generation starts with an empty index of its own.
            rotated = current_generation(client)
            assert rotated is not generation
            assert rotated._key_index == {}
            assert client.get_signing_key("B").key_id == "B"
