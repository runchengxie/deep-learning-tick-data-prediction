import test from 'node:test';
import assert from 'node:assert/strict';
import { replay, scenarios } from '../src/content/order-book-lab.mjs';

test('order lifecycle conserves remaining quantity and aggregates distinct orders', () => {
  const frames = replay(scenarios.lifecycle);
  assert.deepEqual(frames.map(f => f.askTotal), [0, 500, 1000, 600, 100, 300]);
  assert.equal(frames.at(-1).orders.S1.quantity, 100);
  assert.equal(frames.at(-1).orders.S3.quantity, 200);
  assert.equal(frames.at(-1).tradeDelta, 400);
  assert.equal(frames.at(-1).orders.S2, undefined);
});
test('identical final books can hide different trade flow', () => {
  const a = replay(scenarios.pathA).at(-1);
  const b = replay(scenarios.pathB).at(-1);
  assert.equal(a.bidTotal, b.bidTotal);
  assert.equal(a.askTotal, b.askTotal);
  assert.equal(a.tradeDelta, 0);
  assert.equal(b.tradeDelta, -3000);
});
test('adding and withdrawing liquidity changes the book without a trade', () => {
  const frames = replay(scenarios.noTrades);
  assert.equal(frames.at(-1).tradeDelta, 0);
  assert.notEqual(frames[0].bidTotal, frames.at(-1).bidTotal);
});
test('invalid or missing events fail closed; input and earlier states stay immutable', () => {
  assert.throws(() => replay([{type:'trade', id:'missing', quantity:1}]), /unknown/);
  assert.throws(() => replay([{type:'add',id:'x',side:'bid',price:1000,quantity:2},{type:'cancel',id:'x',quantity:3}]), /quantity/);
  assert.throws(() => replay([{type:'add',id:'x',side:'bid',price:1000,quantity:2},{type:'add',id:'x',side:'bid',price:1000,quantity:2}]), /duplicate/);
  const before = JSON.stringify(scenarios.lifecycle);
  const frames = replay(scenarios.lifecycle);
  frames.at(-1).orders.S1.quantity = 99;
  assert.equal(frames[1].orders.S1.quantity,500);
  assert.equal(JSON.stringify(scenarios.lifecycle),before);
});
