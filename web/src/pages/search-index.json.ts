import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { publicDocuments, documentRoute } from '../content/public-documents.mjs';

export async function GET() {
  const root = path.resolve(process.cwd(), '..');
  const entries = await Promise.all(publicDocuments.map(async (source) => {
    const markdown = await readFile(path.resolve(root, source), 'utf8');
    const title = markdown.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? source;
    return { title, route: `${import.meta.env.BASE_URL}${documentRoute(source).slice(1)}`, source, text: markdown.replace(/```[\s\S]*?```/g, ' ').replace(/[`*_>#|]/g, ' ').replace(/\s+/g, ' ').slice(0, 5000) };
  }));
  return new Response(JSON.stringify(entries), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
