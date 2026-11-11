"""Independent PyJWKClient security checks, real RSA signatures, deterministic clock/barriers."""
import io,json,threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.error import URLError
import jwt
import jwt.jwks_client as client_module
import jwt.jwk_set_cache as cache_module
import jwt.api_jwk as jwk_module
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
import pytest

class Clock:
    def __init__(self):self.value=1000.0
    def monotonic(self):return self.value
    def advance(self,seconds):self.value+=seconds

class Endpoint:
    def __init__(self):self.payload={'keys':[]};self.calls=0;self.lock=threading.Lock();self.on_open=None
    def open(self,request,timeout=None):
        with self.lock:
            self.calls+=1;number=self.calls;payload=self.payload
            if not isinstance(payload,Exception):payload=json.loads(json.dumps(payload))
        if self.on_open:self.on_open(number)
        if isinstance(payload,Exception):raise payload
        return io.BytesIO(json.dumps(payload).encode())

@pytest.fixture(scope='module')
def signing_material():
    result={}
    for name,kid in [('A','A'),('B','B'),('A2','A')]:
        private=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        public=json.loads(RSAAlgorithm.to_jwk(private.public_key()));public.update(kid=kid,use='sig',alg='RS256')
        token=jwt.encode({'sub':name},private,algorithm='RS256',headers={'kid':kid})
        result[name]={'private':private,'jwk':public,'token':token}
    return result

@pytest.fixture
def state(monkeypatch):
    clock=Clock();endpoint=Endpoint()
    for module in (client_module,cache_module,jwk_module):
        if hasattr(module,'time'):monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(client_module.urllib.request,'build_opener',lambda *args:endpoint)
    return clock,endpoint

def view(material,*names):return {'keys':[material[name]['jwk'] for name in names]}
def client(**kw):return jwt.PyJWKClient('https://keys.example.invalid/jwks',cache_keys=True,lifespan=120,cooldown_duration=30,**kw)
def validate(instance,token):
    key=instance.get_signing_key_from_jwt(token)
    return jwt.decode(token,key.key,algorithms=['RS256'])

@pytest.mark.parametrize('entrypoint',['fetch_data','get_jwk_set'])
def test_retired_key_invalidated_in_signature_chain(state,signing_material,entrypoint):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client()
    assert validate(c,m['A']['token'])['sub']=='A'
    endpoint.payload=view(m,'B')
    c.fetch_data() if entrypoint=='fetch_data' else c.get_jwk_set(refresh=True)
    with pytest.raises(jwt.PyJWKClientError):validate(c,m['A']['token'])
    assert validate(c,m['B']['token'])['sub']=='B'
    assert endpoint.calls==2
    print({'entrypoint':entrypoint,'retired_A':'rejected','remaining_B':'signature verified','fetches':endpoint.calls})

@pytest.mark.parametrize('entrypoint',['fetch_data','get_jwk_set'])
def test_same_kid_rotation_replaces_cached_object(state,signing_material,entrypoint):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();old=c.get_signing_key('A')
    endpoint.payload=view(m,'A2','B');c.fetch_data() if entrypoint=='fetch_data' else c.get_jwk_set(refresh=True)
    fresh=c.get_signing_key('A');assert fresh is not old
    assert fresh.key.public_numbers()==m['A2']['private'].public_key().public_numbers()
    with pytest.raises(jwt.InvalidSignatureError):validate(c,m['A']['token'])
    assert validate(c,m['A2']['token'])['sub']=='A2'
    assert validate(c,m['B']['token'])['sub']=='B'

@pytest.mark.parametrize('bad',[URLError('local simulated unavailable'),None,[],{'keys':[]},{'keys':[{'kty':'not-a-real-type','kid':'A'}]}],ids=['download-error','null','array','empty-keyset','unusable-key'])
def test_failed_refresh_does_not_publish_or_extend_ttl(state,signing_material,bad):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();assert validate(c,m['A']['token'])['sub']=='A'
    clock.advance(90);endpoint.payload=bad
    with pytest.raises(Exception):c.fetch_data()
    before=endpoint.calls
    assert validate(c,m['A']['token'])['sub']=='A';assert endpoint.calls==before
    clock.advance(31)
    with pytest.raises(Exception):validate(c,m['A']['token'])
    assert endpoint.calls==before+1
    print({'invalid_response':type(bad).__name__,'old_view_valid_before_TTL':True,'old_view_rejected_after_TTL':True})

def test_ttl_expiry_refresh_retires_key(state,signing_material):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();c.get_signing_key('A')
    endpoint.payload=view(m,'B');clock.advance(121)
    with pytest.raises(jwt.PyJWKClientError):c.get_signing_key('A')
    assert validate(c,m['B']['token'])['sub']=='B';assert endpoint.calls==2

def test_unknown_kid_cooldown_prevents_random_id_downloads(state,signing_material):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();c.get_signing_key('A')
    for index in range(20):
        with pytest.raises(jwt.PyJWKClientError):c.get_signing_key('random-'+str(index))
    assert endpoint.calls==1
    clock.advance(31);endpoint.payload=view(m,'B')
    with pytest.raises(jwt.PyJWKClientError):c.get_signing_key('random-next')
    for index in range(20):
        with pytest.raises(jwt.PyJWKClientError):c.get_signing_key('another-'+str(index))
    assert endpoint.calls==2
    with pytest.raises(jwt.PyJWKClientError):c.get_signing_key('A')
    assert validate(c,m['B']['token'])['sub']=='B'

@pytest.mark.parametrize('mode',['expired-known','unknown-cooldown'])
def test_two_threads_share_coherent_refresh(state,signing_material,mode):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();c.get_signing_key('A')
    clock.advance(121 if mode=='expired-known' else 31);endpoint.payload=view(m,'A2','B')
    entered=threading.Event();release=threading.Event();start=threading.Barrier(3)
    def hold(number):
        if number==2:
            entered.set();assert release.wait(8),'verification barrier was not released'
    endpoint.on_open=hold
    def worker(index):
        start.wait(timeout=8)
        if mode=='unknown-cooldown':
            try:c.get_signing_key('unknown-'+str(index))
            except jwt.PyJWKClientError:return 'rejected'
            return 'accepted'
        return validate(c,m['A2']['token'])['sub']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,index) for index in range(2)]
        try:
            start.wait(timeout=8);assert entered.wait(8),'no thread reached refresh'
        finally:release.set()
        results=[future.result(timeout=8) for future in futures]
    assert results==(['rejected','rejected'] if mode=='unknown-cooldown' else ['A2','A2'])
    assert endpoint.calls==2
    assert validate(c,m['B']['token'])['sub']=='B'
    print({'race':mode,'thread_results':results,'fetches':endpoint.calls})

def test_query_overlapping_direct_refresh_returns_complete_generation(state,signing_material):
    clock,endpoint=state;m=signing_material;endpoint.payload=view(m,'A','B');c=client();c.get_signing_key('A')
    entered=threading.Event();release=threading.Event();endpoint.payload=view(m,'A2','B')
    def hold(number):
        if number==2:entered.set();assert release.wait(8)
    endpoint.on_open=hold
    with ThreadPoolExecutor(max_workers=2) as pool:
        refresh=pool.submit(c.fetch_data)
        try:
            assert entered.wait(8)
            lookup=pool.submit(c.get_signing_keys)
        finally:release.set()
        refresh.result(timeout=8);keys=lookup.result(timeout=8)
    actual={key.key_id:key.key.public_numbers() for key in keys}
    assert set(actual)=={'A','B'}
    assert actual['A'] in (m['A']['private'].public_key().public_numbers(),m['A2']['private'].public_key().public_numbers())
    assert actual['B']==m['B']['private'].public_key().public_numbers()
    assert validate(c,m['A2']['token'])['sub']=='A2'
    with pytest.raises(jwt.InvalidSignatureError):validate(c,m['A']['token'])
