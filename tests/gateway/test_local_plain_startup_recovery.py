"""A plain-text local session (desktop chat) must be recoverable at startup without a receipt.

Production failure (2026-10, relay-APK sessions, local-a35e11b7): a desktop chat had a queued
message and an ``unknown`` head left by a hard-killed owner. After a restart the session was
unmountable from every angle — route recovery refuses LOCAL rows, local recovery needs a
worker receipt (the store had none), and ``recover_native_sessions`` raised ``not_found``
because desktop rows carry plain ``{'text'}`` payloads with no ``native_text_v1`` envelope.
Result: ``attach`` answered ``not_found`` forever and the queued backlog never drained.

The repaired contract:
* a local row whose session still names a route mounts at startup and its queued rows drain;
* a dead-epoch ``unknown`` settles (interrupted, never replayed) only when a queued turn is
  blocked behind it — an unblocked ``unknown`` keeps the operator's Discard decision;
* a row without a route stays ``not_found`` (nothing to mount it under).
"""
import asyncio

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


def _local_row(db):
    """The desktop-chat shape: local source, its own dm route, no receipt, no native payload."""
    def write(conn):
        conn.execute("UPDATE sessions SET source='local', session_key=?, chat_id=?, user_id=?, "
                     "origin_json=? WHERE id='s'",
                     ('agent:main:local:dm:s', 's', 'owned',
                      '{"platform":"local","chat_id":"s","user_id":"owned"}'))
    db._execute_write(write)


def _stamp_unknown(db, admission_id):
    """The hard-killed-owner state: claimed turn recovered as unknown (dead epoch)."""
    def write(conn):
        conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (admission_id,))
    db._execute_write(write)


def _bindings(authority):
    source = type('S', (), {'platform': type('P', (), {'value': 'local'})()})()
    return [(REF.session_id, source, None)]


def _executor(seen):
    async def execute(owner, ref, row):
        seen.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': row['request_id'] + ' done', 'completed': True}, 'usage': {}}
        return row['request_id'] + ' done'
    return execute


@pytest.mark.asyncio
async def test_plain_local_queued_drains_after_restart_without_receipts(tmp_path, monkeypatch):
    """Queued desktop message, no receipt, no native envelope: recovery mounts and drains."""
    db, authority = _authority(tmp_path, monkeypatch)
    _local_row(db)
    executed = []
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', _executor(executed))
    with db:
        receipt = await authority.submit(ACTOR, Submission('plain-queued', REF, {'text': 'plain-queued'}, 'queue'))
        # Simulate the restart: the live entry is gone; recovery must rebuild it.
        authority.sessions.pop('s')
        authority.waiters.clear()
        verdicts = await authority.recover_native_sessions(_bindings(authority))
        assert verdicts == {'s': 'ready_local'}, verdicts
        assert 's' in authority.sessions, 'the session was not mounted: attach would answer not_found'
        task = authority.sessions['s'].task
        assert task is not None
        await asyncio.wait_for(task, 10)
        assert executed == ['plain-queued']
        assert get_session_admission(db, admission_id=receipt.admission_id)['status'] == 'terminal'


@pytest.mark.asyncio
async def test_dead_epoch_unknown_settles_when_a_queued_turn_is_blocked(tmp_path, monkeypatch):
    """unknown head + queued follower: startup settles the unknown (never replayed) and drains."""
    db, authority = _authority(tmp_path, monkeypatch)
    _local_row(db)
    executed = []
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', _executor(executed))
    with db:
        head = await authority.submit(ACTOR, Submission('dead-head', REF, {'text': 'dead-head'}, 'queue'))
        follower = await authority.submit(ACTOR, Submission('blocked-follower', REF,
                                                            {'text': 'blocked-follower'}, 'queue'))
        _stamp_unknown(db, head.admission_id)
        authority.sessions.pop('s')
        authority.waiters.clear()
        verdicts = await authority.recover_native_sessions(_bindings(authority))
        assert verdicts == {'s': 'ready_local'}, verdicts
        task = authority.sessions['s'].task
        assert task is not None
        await asyncio.wait_for(task, 10)
        # The dead turn settles terminal without executing; the follower runs exactly once.
        assert executed == ['blocked-follower'], executed
        head_row = get_session_admission(db, admission_id=head.admission_id)
        assert head_row['status'] == 'terminal', head_row
        assert head_row['outcome'] in ('interrupted', 'discarded'), head_row
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'terminal'


@pytest.mark.asyncio
async def test_unblocked_unknown_stays_paused_but_the_session_opens(tmp_path, monkeypatch):
    """No queued behind it: the Discard decision belongs to the operator — but the session
    must mount, so desktop can open it instead of answering not_found."""
    db, authority = _authority(tmp_path, monkeypatch)
    _local_row(db)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', _executor([]))
    with db:
        head = await authority.submit(ACTOR, Submission('lonely-unknown', REF, {'text': 'lonely-unknown'}, 'queue'))
        _stamp_unknown(db, head.admission_id)
        authority.sessions.pop('s')
        authority.waiters.clear()
        verdicts = await authority.recover_native_sessions(_bindings(authority))
        assert verdicts == {'s': 'paused'}, verdicts
        assert 's' in authority.sessions, 'paused must still mean mountable: open must not be not_found'
        assert get_session_admission(db, admission_id=head.admission_id)['status'] == 'unknown'


@pytest.mark.asyncio
async def test_no_route_local_row_stays_not_found(tmp_path, monkeypatch):
    """Without a session_key there is nothing to mount it under: the honest not_found stays."""
    db, authority = _authority(tmp_path, monkeypatch)
    # source local but NO session_key
    def write(conn):
        conn.execute("UPDATE sessions SET source='local', chat_id='s' WHERE id='s'")
    db._execute_write(write)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', _executor([]))
    with db:
        await authority.submit(ACTOR, Submission('orphan-text', REF, {'text': 'orphan-text'}, 'queue'))
        authority.sessions.pop('s')
        authority.waiters.clear()
        verdicts = await authority.recover_native_sessions(_bindings(authority))
        assert verdicts == {'s': 'not_found'}, verdicts
