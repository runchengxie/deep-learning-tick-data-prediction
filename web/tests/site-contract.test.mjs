import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { publicDocuments, documentRoute, documentSlug } from '../src/content/public-documents.mjs';
import { studies, evidenceCutoff } from '../src/content/studies.mjs';

const root = resolve(import.meta.dirname, '../..');

test('public documentation is an explicit, valid source allowlist', () => {
  assert.equal(new Set(publicDocuments).size, publicDocuments.length);
  assert.ok(publicDocuments.length >= 40);
  for (const source of publicDocuments) {
    assert.ok(source.startsWith('docs/') && source.endsWith('.md'));
    assert.ok(!source.includes('/superpowers/') && !source.endsWith('.pdf'));
    assert.ok(existsSync(resolve(root, source)), `missing ${source}`);
    assert.ok(documentRoute(source).startsWith('/'));
    assert.ok(!documentSlug(source).includes('..'));
  }
});

test('site offers persistent Chinese and dark-mode preferences', () => {
  const layout = readFileSync(resolve(root, 'web/src/layouts/SiteLayout.astro'), 'utf8');
  const styles = readFileSync(resolve(root, 'web/src/styles/global.css'), 'utf8');
  const home = readFileSync(resolve(root, 'web/src/pages/index.astro'), 'utf8');
  const detail = readFileSync(resolve(root, 'web/src/pages/studies/[id].astro'), 'utf8');
  const article = readFileSync(resolve(root, 'web/src/layouts/ArticleLayout.astro'), 'utf8');

  assert.match(layout, /data-locale-toggle/);
  assert.match(layout, /data-theme-toggle/);
  assert.match(layout, /qdl-locale/);
  assert.match(layout, /qdl-theme/);
  assert.match(home, /data-i18n-zh=/);
  assert.match(detail, /data-i18n-zh=/);
  assert.match(article, /currently available in English/);
  assert.match(styles, /:root\[data-theme="dark"\]/);
  assert.match(layout, /prefers-color-scheme: dark/);
  assert.match(layout, /localStorage\.setItem\('qdl-locale'/);
  assert.match(layout, /localStorage\.setItem\('qdl-theme'/);
  for (const study of studies) {
    assert.ok(study.zh?.title && study.zh?.summary && study.zh?.openQuestion);
    assert.equal(study.zh.values.length, study.values.length);
  }
});

test('research summaries retain evidence provenance and distinguish ranking from returns', () => {
  assert.match(evidenceCutoff, /^\d{4}-\d{2}-\d{2}$/);
  assert.equal(new Set(studies.map((study) => study.id)).size, studies.length);
  for (const study of studies) {
    assert.ok(publicDocuments.includes(study.source));
    assert.ok(study.stage && study.sample && study.openQuestion && study.chartDescription);
    assert.ok(study.values.length > 0 && study.values.every((value) => value.value >= 0));
  }
  const status = readFileSync(resolve(root, 'docs/project-status.md'), 'utf8');
  assert.match(status, /negative net active return|no viable region/i);
});
