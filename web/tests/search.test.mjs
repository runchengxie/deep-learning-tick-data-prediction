import test from 'node:test';
import assert from 'node:assert/strict';
import { searchEntries, documentSearchText } from '../src/content/search.mjs';

test('Chinese aliases match English records and all terms remain required', () => {
  const index = [{ title: 'Event-stream cost study', text: 'order book' }, { title: 'Book', text: 'order book' }];
  assert.equal(searchEntries(index, '事件流 成本').length, 1);
  assert.equal(searchEntries(index, '订单簿').length, 2);
  assert.equal(searchEntries(index, '事件流 nonexistent').length, 0);
  assert.equal(searchEntries(index, '').length, 0);
});
test('long public documents remain searchable beyond the old truncation boundary', () => {
  const text = documentSearchText('# Heading\n' + 'context '.repeat(1000) + '\nlate-evidence');
  assert.equal(searchEntries([{title:'Long record',text}], 'late-evidence').length, 1);
});
