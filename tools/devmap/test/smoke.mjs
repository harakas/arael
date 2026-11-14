// A browser smoke test of the viewer: every case and view in a headless
// Chromium, light and dark; page errors and the page's error strip are
// reported, and a screenshot of each view is written.
//
//   test/smoke.sh [--out DIR] [--cases a,b] [--shots all|first|none]
//
// Needs Node and Playwright with its Chromium (smoke.sh says where it
// looks). Exit code 1 when anything threw.
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const require = createRequire(path.join(process.env.DEVMAP_PLAYWRIGHT || process.cwd(), 'noop.js'));
const { chromium } = require('playwright');

const here = path.dirname(fileURLToPath(import.meta.url));
const page_url = pathToFileURL(path.join(here, '..', 'index.html')).href;
const arg = (name, dflt) => {
    const i = process.argv.indexOf('--' + name);
    return i > 0 ? process.argv[i + 1] : dflt;
};
const out = arg('out', path.join(here, 'shots'));
const shots = arg('shots', 'first');
const only = arg('cases', '');
fs.mkdirSync(out, { recursive: true });

const views = ['overview', 'pipeline', 'matrix', 'memory', 'kernels', 'findings'];
const problems = [];
const browser = await chromium.launch();

for (const scheme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width: 1680, height: 1000 }, colorScheme: scheme });
    page.on('pageerror', e => problems.push(`[${scheme}] page error: ${e.message}`));
    // the generated case list is missing before the first gen/run.sh: the viewer then shows the sample
    const optional = m => (m.location().url || '').endsWith('/data/cases.js');
    page.on('console', m => { if (m.type() === 'error' && !optional(m)) problems.push(`[${scheme}] console error: ${m.text()}`); });
    await page.goto(page_url);
    await page.waitForFunction(() => window.DM && DM.state && DM.state.caseId, null, { timeout: 20000 });
    let cases = await page.evaluate(() => DEVMAP.cases.map(c => c.id));
    if (only) cases = cases.filter(c => only.split(',').includes(c));
    const strip = async where => {
        const text = await page.evaluate(() => { const e = document.getElementById('errors'); return e && !e.hidden ? e.innerText.trim() : ''; });
        if (text) problems.push(`[${scheme}] ${where}: error strip: ${text.slice(0, 400)}`);
    };
    await strip('after load');
    for (const [ci, id] of cases.entries()) {
        await page.evaluate(c => DM.setCase(c), id);
        // a case without measured accesses says so in the notice, and only such a case;
        // its amounts read n/m, never a question mark
        const seen = await page.evaluate(() => {
            const c = DEVMAP.cases.find(c => c.id === DM.state.caseId);
            const launches = c.phases.flatMap(p => p.steps).filter(s => s.kind === 'launch');
            return { launches: launches.length, measured: launches.some(s => (s.access || []).some(a => a.src === 'measured')),
                     notice: !document.getElementById('notice').hidden, from: (c.meta || {}).patterns_from || '',
                     link: !!document.querySelector('#notice .case-link') };
        });
        if (seen.launches && seen.notice === seen.measured) problems.push(`${id}: the notice is ${seen.notice ? 'shown' : 'hidden'}, measured accesses: ${seen.measured}`);
        if (seen.from && cases.includes(seen.from) && !seen.link) problems.push(`${id}: the notice does not link the probed case ${seen.from}`);
        // the Memory table has a computable cell for every index section and for no other
        await page.evaluate(() => DM.setView('memory'));
        await page.waitForTimeout(100);
        const cmp = await page.evaluate(() => {
            const c = DEVMAP.cases.find(c => c.id === DM.state.caseId);
            const index = c.blocks.flatMap(b => b.sections).filter(s => s.content).length;
            const cells = [...document.querySelectorAll('#view-body td.mem-td-cmp')];
            return { index, filled: cells.filter(td => td.textContent.trim()).length, head: !!document.querySelector('#view-body th[data-tip^="index sections"]'),
                     whole: c.blocks.flatMap(b => b.sections).filter(s => s.computable === 1).length, yes: document.querySelectorAll('#view-body td.mem-td-cmp .cmp-yes').length };
        });
        if (!cmp.head || cmp.filled !== cmp.index || cmp.yes !== cmp.whole) problems.push(`${id}: the computable column: ${JSON.stringify(cmp)}`);
        // an index section's kept words are shown in its details: one section of each kind of content
        const sampled = await page.evaluate(() => {
            const c = DEVMAP.cases.find(c => c.id === DM.state.caseId);
            const seen = {}, out = [];
            for (const b of c.blocks) for (const s of b.sections) {
                if (!s.sample || !s.sample.head || !s.sample.head.length) continue;
                const kind = ((s.content || {}).kind || 'none') + (s.record > 1 ? '-declared' : '');
                if (seen[kind]) continue;
                seen[kind] = true;
                out.push({ sel: `section:${b.id}:${s.i}`, name: b.name + '.' + s.name, kind, first: String(s.sample.head[0]) });
            }
            return out;
        });
        const wanted = shots === 'all' || (shots === 'first' && (ci === 0 || id === 'sample' || id.includes('slam-120-f32') || id.includes('372-f32')));
        for (const s of sampled) {
            await page.evaluate(x => DM.select(x), s.sel);
            await page.waitForTimeout(50);
            const cell = await page.evaluate(() => { const td = document.querySelector('#det-body table.vals tr:not(.vals-head) td'); return td ? td.textContent : null; });
            if (cell !== s.first) problems.push(`${id}: ${s.name} (${s.kind}): the values grid starts with ${cell}, the sample with ${s.first}`);
            await strip(`${id} values of ${s.name}`);
            if (wanted) await page.screenshot({ path: path.join(out, `${id}-values-${s.kind}-${scheme}.png`) });
        }
        for (const view of views) {
            await page.evaluate(v => DM.setView(v), view);
            await page.waitForTimeout(150);
            await strip(`${id} ${view}`);
            const marks = await page.evaluate(() => [...document.querySelectorAll('#view-body *')]
                .filter(e => !e.children.length && /^(\+?\?|[RW] \?)$/.test(e.textContent.trim())).length);
            if (marks) problems.push(`${id} ${view}: ${marks} question marks for unmeasured amounts`);
            // select the first few things the view offers, as a user would click them
            const sels = await page.evaluate(() => {
                const c = DEVMAP.cases.find(c => c.id === DM.state.caseId);
                const steps = c.phases.flatMap(p => p.steps).slice(0, 400);
                const launch = steps.find(s => s.kind === 'launch' && s.access && s.access.length);
                const b = c.blocks[c.blocks.length - 1];
                return [launch && 'step:' + launch.id, launch && 'kernel:' + launch.kernel, b && `section:${b.id}:${Math.min(1, b.sections.length - 1)}`].filter(Boolean);
            });
            for (const s of sels) {
                await page.evaluate(x => DM.select(x), s);
                await strip(`${id} ${view} select ${s}`);
            }
            if (shots === 'all' || (shots === 'first' && (ci === 0 || id.includes('slam-120-f32') || id.includes('372-f32')))) {
                await page.screenshot({ path: path.join(out, `${id}-${view}-${scheme}.png`) });
            }
        }
    }
    await page.close();
}
await browser.close();
const unique = [...new Set(problems)];
console.log(unique.length ? unique.join('\n') : 'no errors');
console.log(`screenshots in ${out}`);
process.exit(unique.length ? 1 : 0);
