export const aliases = {
  '事件流': ['event stream', 'event-stream'], '成本': ['cost'],
  '订单簿': ['order book', 'order-book'], '盘口': ['snapshot', 'order book'],
  '逐笔委托': ['order messages'], '逐笔成交': ['trade messages'],
  '订单流': ['order flow'], '撤单': ['cancel'], '分钟': ['minute'],
  '排序': ['ranking', 'rank'], '深市': ['shenzhen'], '沪市': ['shanghai'],
};

export function searchEntries(index, query) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return [];
  return index.filter(item => {
    const content = `${item.title} ${item.text}`.toLowerCase();
    return terms.every(term => [term, ...(aliases[term] ?? [])].some(candidate => content.includes(candidate)));
  });
}

export function documentSearchText(markdown) {
  return markdown.replace(/```[\s\S]*?```/g, ' ').replace(/[`*_>#|]/g, ' ').replace(/\s+/g, ' ').trim();
}

export function catalogSearchText(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(catalogSearchText).join(' ');
  if (value && typeof value === 'object') return Object.values(value).map(catalogSearchText).join(' ');
  return '';
}
