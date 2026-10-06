import path from 'node:path';
import { publicDocuments, documentRoute } from './public-documents.mjs';

const publicSources = new Set(publicDocuments);
const repository = 'https://github.com/runchengxie/quant-deep-learning/blob/main';

export function rewritePublicLinks({ source, base }) {
  const sourceDirectory = path.posix.dirname(source);
  return (tree) => {
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'element' && node.tagName === 'a' && typeof node.properties?.href === 'string') {
        const href = node.properties.href;
        if (!/^(?:[a-z]+:|\/\/|#|data:|javascript:|mailto:)/i.test(href)) {
          const [rawPath, fragment] = href.split('#', 2);
          if (rawPath) {
            const target = path.posix.normalize(path.posix.join(sourceDirectory, rawPath));
            if (publicSources.has(target)) {
              node.properties.href = `${base}${documentRoute(target)}${fragment ? `#${fragment}` : ''}`;
            } else if (!target.startsWith('../') && !target.startsWith('/')) {
              node.properties.href = `${repository}/${target}${fragment ? `#${fragment}` : ''}`;
            }
          }
        }
      }
      for (const child of node.children ?? []) visit(child);
    };
    visit(tree);
  };
}
