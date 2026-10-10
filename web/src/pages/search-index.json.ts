import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { publicDocuments, documentRoute } from '../content/public-documents.mjs';
import { documentSearchText, catalogSearchText } from '../content/search.mjs';
import { studies } from '../content/studies.mjs';
import { studyReports } from '../content/study-reports.mjs';
import { copy } from '../content/market-concepts.mjs';

export async function GET() {
  const root = path.resolve(process.cwd(), '..');
  const entries = await Promise.all(publicDocuments.map(async (source) => {
    const markdown = await readFile(path.resolve(root, source), 'utf8');
    const title = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? source;
    return { title, route: `${import.meta.env.BASE_URL}${documentRoute(source).slice(1)}`, source, text: documentSearchText(markdown) };
  }));
  for (const study of studies) entries.push({ title: `${study.title} / ${study.zh.title}`, route: `${import.meta.env.BASE_URL}studies/${study.id}/`, source: 'Research study', text: catalogSearchText([study, studyReports[study.id as keyof typeof studyReports]]) });
  entries.push({ title: copy.title.join(' / '), route: `${import.meta.env.BASE_URL}concepts/order-book/`, source: 'Interactive concepts', text: Object.values(copy).flat().join(' ') });
  entries.push({ title: copy.sectionName.join(' / '), route: `${import.meta.env.BASE_URL}concepts/`, source: 'Concepts index', text: Object.values(copy).flat().join(' ') });
  return new Response(JSON.stringify(entries), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
