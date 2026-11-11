"""A separate verification console for real signed JWTs and deterministic JWKS updates."""
import argparse,importlib.util,traceback
from pathlib import Path
import jwt,pytest
p=argparse.ArgumentParser();p.add_argument('--scenario',choices=['retired','replaced','expiry','concurrent'],required=True);a=p.parse_args()
path=Path(__file__).with_name('test_acceptance.py');spec=importlib.util.spec_from_file_location('key_view_checks',path);checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
print('Library: PyJWT '+jwt.__version__)
print('JWKS source: in-process local transport; no external endpoint')
print('Signatures: generated RSA keys, RS256 verification')
try:
    with pytest.MonkeyPatch.context() as patch:
        state=checks.state.__wrapped__(patch);material=checks.signing_material.__wrapped__()
        if a.scenario=='retired':
            checks.test_retired_key_invalidated_in_signature_chain(state,material,'fetch_data')
            print('Old key A: refused after the accepted refresh')
            print('Remaining key B: signature accepted')
        elif a.scenario=='replaced':
            checks.test_same_kid_rotation_replaces_cached_object(state,material,'fetch_data')
            print('Same key ID, new public key: selected')
            print('Old signature: rejected; new signature: accepted')
        elif a.scenario=='expiry':
            checks.test_failed_refresh_does_not_publish_or_extend_ttl(state,material,checks.URLError('local unavailable'))
            print('Failed refresh: old accepted view available within its original TTL')
            print('After expiry: stale key refused')
        else:
            checks.test_two_threads_share_coherent_refresh(state,material,'expired-known')
            print('Two waiting queries: both verified with the replacement key')
            print('Refresh requests: one after the initial accepted view')
    print('Observed check: PASS')
except Exception:
    traceback.print_exc();print('Observed check: FAIL');raise SystemExit(1)
