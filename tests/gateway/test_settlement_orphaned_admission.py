"""A settlement that finds its admission gone (row deleted, no tombstone) must not stall the FIFO.

Production failure (2026-10, relay-APK sessions): ``finish_result`` read an admission that no
longer existed, ``RuntimeStoreError('not_found')`` escaped settlement, the recovery stamp hit the
same missing row, and the drain left the session paused ``unknown_execution`` with no row left to
resolve — every later open of that session reported ``not_found``. These scenarios pin the
repaired contract: an orphaned claim is released (its viewers get a terminal frame, the FIFO
keeps moving) instead of turning the whole session into a permanent error.
"""
import asyncio

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import get_session_admission
from hermes_state_terminal import ADMISSION_PREFIX
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


def _submit(authority, request_id):
    return authority.submit(ACTOR, Submission(request_id, REF, {'text': request_id}, 'queue'))


def _orphan(db, admission_id):
    """The orphan state: the admission row is deleted and no tombstone exists for it.

    Reached in the field by a retirement that raced the claim or by tombstone retention sweeping
    a receipt whose owner still held the claim in memory. Both look identical to settlement.
    """
    def write(conn):
        conn.execute('DELETE FROM session_admissions WHERE admission_id=?', (admission_id,))
        conn.execute('DELETE FROM state_meta WHERE key=?', (ADMISSION_PREFIX + admission_id,))
    return db._execute_write(write)


def _executor(owner_seen):
    async def execute(owner, ref, row):
        owner_seen.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': row['request_id'] + ' done', 'completed': True}, 'usage': {}}
        return row['request_id'] + ' done'
    return execute


@pytest.mark.asyncio
async def test_orphaned_admission_settlement_releases_fifo(tmp_path, monkeypatch):
    """The row vanishes while the turn runs (the logged ``finish_result`` traceback).

    The drain must survive, the producing owner must forget its pending result, the orphaned
    turn's viewers must get a terminal completion frame (not silence), and the queued follower
    must still run: a missing receipt row is not a reason to park the session forever.
    """
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    from gateway import session_settlement_recovery as recovery
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01, 0.01), raising=False)
    calls, frames = [], []
    authority.sessions['s'].event_stream.observers.add(frames.append)

    async def execute(owner, ref, row):
        calls.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': row['request_id'] + ' done', 'completed': True}, 'usage': {}}
        if row['request_id'] == 'orphaned-head':
            _orphan(db, row['admission_id'])
        return row['request_id'] + ' done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

    with db:
        head = await _submit(authority, 'orphaned-head')
        follower = await _submit(authority, 'follower-after-orphan')
        await asyncio.wait_for(authority._drain(REF), 10)

        assert calls == ['orphaned-head', 'follower-after-orphan']
        assert head.admission_id not in authority.pending_results
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'terminal'
        completes = [f for f in frames if f.get('params', {}).get('type') == 'message.complete']
        head_frame = [f for f in completes if f['params'].get('payload', {}).get('text') == 'orphaned-head done']
        assert head_frame, f'no terminal frame for the orphaned turn: {frames!r}'
        # An honest failure notice: the answer is shown, but nothing durable committed it.
        assert head_frame[0]['params']['payload']['outcome'] == 'failed'


@pytest.mark.asyncio
async def test_orphaned_admission_during_unknown_recovery_keeps_fifo_moving(tmp_path, monkeypatch):
    """The row vanishes after a transient settlement failure, as the unknown stamp goes in.

    Production shape: storage faults out every settle retry, ``_mark_unknown`` then finds no row.
    Pre-repair the drain paused ``unknown_execution`` and returned — with no row left, nothing
    could ever resolve it and the session died. The follower must still run.
    """
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    from gateway import session_results, session_settlement_recovery as recovery
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01, 0.01), raising=False)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', _executor([]))
    original_finish = session_results.finish_result
    original_mark = recovery._mark_unknown

    def storage_down(db_, *, epoch, row, response, outcome, result=None, prepared=None, transfers=None):
        # Only the head's settlement faults; the follower must settle normally afterwards.
        if row['request_id'] == 'lost-during-recovery':
            raise OSError('settlement storage unavailable')
        return original_finish(db_, epoch=epoch, row=row, response=response, outcome=outcome,
                               result=result, prepared=prepared, transfers=transfers)

    def mark_after_orphan(authority_, row):
        if row['request_id'] == 'lost-during-recovery':
            _orphan(db, row['admission_id'])
        return original_mark(authority_, row)

    monkeypatch.setattr(session_results, 'finish_result', storage_down)
    monkeypatch.setattr(recovery, '_mark_unknown', mark_after_orphan)

    with db:
        await _submit(authority, 'lost-during-recovery')
        follower = await _submit(authority, 'follower-parked')
        await asyncio.wait_for(authority._drain(REF), 10)
        # Whatever the head became, the FIFO moved on: a vanished row cannot park a session.
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'terminal'
