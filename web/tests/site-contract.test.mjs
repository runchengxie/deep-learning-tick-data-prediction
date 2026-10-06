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
