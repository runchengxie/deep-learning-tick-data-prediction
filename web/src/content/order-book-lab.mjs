const add = (id, side, quantity) => ({type:'add', id, side, price:side === 'bid' ? 999 : 1001, quantity});
const initial = [add('B0','bid',10000), add('A0','ask',10000)];
export const scenarios = {
  lifecycle: [add('S1','ask',500), add('S2','ask',500), {type:'trade',id:'S1',quantity:400}, {type:'cancel',id:'S2',quantity:500}, add('S3','ask',200)],
  pathA: [...initial, add('B1','bid',5000), {type:'cancel',id:'B1',quantity:5000}],
  pathB: [...initial, {type:'trade',id:'B0',quantity:3000}, add('B2','bid',3000)],
  noTrades: [...initial, add('B1','bid',8000), {type:'cancel',id:'B1',quantity:8000}, add('B2','bid',3000)],
};
export function replay(events) {
  const orders = {};
  let tradeDelta = 0;
  const snapshot = () => {
    const rows = Object.values(orders);
    const bidTotal = rows.filter(o=>o.side==='bid').reduce((s,o)=>s+o.quantity,0);
    const askTotal = rows.filter(o=>o.side==='ask').reduce((s,o)=>s+o.quantity,0);
    return {orders:structuredClone(orders), bidTotal, askTotal, tradeDelta, obi:bidTotal+askTotal ? (bidTotal-askTotal)/(bidTotal+askTotal) : null};
  };
  const frames = [snapshot()];
  for (const event of events) {
    if (!Number.isSafeInteger(event.quantity) || event.quantity <= 0) throw new Error('invalid quantity');
    if (event.type === 'add') {
      if (orders[event.id]) throw new Error('duplicate order');
      if (!['bid','ask'].includes(event.side) || !Number.isSafeInteger(event.price) || event.price <= 0 || !event.id) throw new Error('invalid order');
      orders[event.id] = {id:event.id, side:event.side, price:event.price, quantity:event.quantity};
    } else if (event.type === 'trade' || event.type === 'cancel') {
      const order = orders[event.id];
      if (!order) throw new Error('unknown order');
      if (event.quantity > order.quantity) throw new Error('quantity exceeds remaining order');
      if (event.type === 'trade') tradeDelta += order.side === 'ask' ? event.quantity : -event.quantity;
      order.quantity -= event.quantity;
      if (!order.quantity) delete orders[event.id];
    } else throw new Error('unknown event type');
    frames.push(snapshot());
  }
  return frames;
}
