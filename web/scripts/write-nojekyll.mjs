import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const output = path.resolve(process.cwd(), 'dist');
await mkdir(output, { recursive: true });
await writeFile(path.join(output, '.nojekyll'), '');
