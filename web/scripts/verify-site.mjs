import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { publicDocuments, documentSlug } from '../src/content/public-documents.mjs';
import { studies } from '../src/content/studies.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.resolve(process.argv[2] ?? path.join(root, 'dist'));
const base = '/quant-deep-learning';
const failures = [];
const expected = [
  'index.html', '404.html', 'documentation/index.html', 'studies/index.html',
  'concepts/index.html', 'concepts/order-book/index.html',
  'search/index.html', 'search-index.json', 'favicon.svg',
  ...studies.map((study) => `studies/${study.id}/index.html`),
  ...publicDocuments.map((source) => `${documentSlug(source)}/index.html`),
];
for (const relative of new Set(expected)) {
  if (!fs.existsSync(path.join(dist, relative))) failures.push(`missing public route/artifact: ${relative}`);
}
for (const file of publicDocuments) {
  if (!fs.existsSync(path.resolve(root, '..', file))) failures.push(`missing allowlisted source: ${file}`);
  if (file.includes('/superpowers/') || file.endsWith('.pdf')) failures.push(`private/internal source is allowlisted: ${file}`);
}
const filesBelow = (directory) => fs.existsSync(directory) ? fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
  const target = path.join(directory, entry.name);
  return entry.isDirectory() ? filesBelow(target) : [target];
}) : [];
const files = filesBelow(dist);
const htmlFiles = files.filter((file) => file.endsWith('.html'));
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8');
  const relative = path.relative(dist, file).split(path.sep).join('/');
  if (!html.includes('<html lang="en">')) failures.push(`${relative}: missing English document language`);
  for (const [, href] of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    if (/^(?:[a-z]+:|\/\/|#|data:|javascript:|mailto:)/i.test(href)) continue;
    const targetPath = href.split(/[?#]/, 1)[0];
    if (targetPath.startsWith('/')) {
      if (targetPath !== base && !targetPath.startsWith(`${base}/`)) failures.push(`${relative}: local URL outside Pages base: ${href}`);
      else {
        const local = path.join(dist, targetPath.slice(base.length).replace(/^\//, ''));
        const target = [local, path.join(local, 'index.html')].find((candidate) => fs.existsSync(candidate));
        if (!target) failures.push(`${relative}: unresolved local URL ${href}`);
        else if (href.includes('#')) {
          const fragment = decodeURIComponent(href.split('#').at(-1));
          const targetFile = target.endsWith('.html') ? target : path.join(target, 'index.html');
          const targetHtml = fs.existsSync(targetFile) ? fs.readFileSync(targetFile, 'utf8') : '';
          if (!new RegExp(`(?:id|name)="${fragment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`).test(targetHtml)) {
            failures.push(`${relative}: unresolved heading anchor ${href}`);
          }
        }
      }
    }
  }
}
const publicFiles = files.filter((file) => /\.(?:html|js|css|json|txt|svg|xml)$/i.test(file));
const publicText = publicFiles.map((file) => fs.readFileSync(file, 'utf8')).join('\n');
for (const marker of [/\/home\//, /\/Users\//, /\/mnt\//, /TUSHARE_TOKEN=/, /API_KEY=/, /SECRET_KEY=/, /-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----/]) {
  if (marker.test(publicText)) failures.push(`public output contains forbidden marker: ${marker}`);
}
if (!fs.existsSync(path.join(dist, '.nojekyll'))) failures.push('missing .nojekyll');
if (failures.length) {
  console.error(failures.map((failure) => `- ${failure}`).join('\n'));
  process.exit(1);
}
console.log(`Verified ${new Set(expected).size} public routes/artifacts and ${publicDocuments.length} allowlisted Markdown sources.`);
