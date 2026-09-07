#!/usr/bin/env npx tsx
/**
 * Falsification experiment for rule-load-order invariance.
 *
 * This is a measurement harness, not a gate. It compares matched rule-ID sets
 * for the same rules, sample, and production shape while changing only rule
 * order. A confirmed mismatch is printed as JSON and exits 2.
 */
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ATREngine } from '../src/engine.js';
import { corpusShapes } from '../src/corpus-event.js';
import { loadRulesFromDirectory } from '../src/loader.js';
import type { AgentEvent, ATRRule } from '../src/types.js';

interface Sample {
  readonly id: string;
  readonly text: string;
  readonly sources: readonly string[];
}

interface EvaluationCase {
  readonly key: string;
  readonly sampleId: string;
  readonly text: string;
  readonly shape: string;
  readonly event?: AgentEvent;
}

interface Order {
  readonly name: string;
  readonly rules: readonly ATRRule[];
}

interface Mismatch {
  readonly caseKey: string;
  readonly sampleId: string;
  readonly shape: string;
  readonly order: string;
  readonly baseline: readonly string[];
  readonly actual: readonly string[];
  readonly missing: readonly string[];
  readonly unexpected: readonly string[];
}

function valueAfter(argv: readonly string[], flag: string): string | undefined {
  const at = argv.indexOf(flag);
  if (at === -1) return undefined;
  const value = argv[at + 1];
  if (value === undefined || value.startsWith('--')) throw new Error(`${flag} requires a value`);
  return value;
}

function ids(matches: readonly { readonly rule: { readonly id: string } }[]): string[] {
  return [...new Set(matches.map((match) => match.rule.id))].sort();
}

function sameIds(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

function evaluateCase(engine: ATREngine, item: EvaluationCase): string[] {
  return item.event === undefined ? ids(engine.scanSkill(item.text)) : ids(engine.evaluate(item.event));
}

async function engineFor(rules: readonly ATRRule[]): Promise<ATREngine> {
  const engine = new ATREngine({ rules: [...rules], rulesDir: resolve('.missing-order-experiment-rules') });
  const loaded = await engine.loadRules();
  if (loaded !== rules.length) throw new Error(`engine loaded ${loaded}/${rules.length} rules`);
  return engine;
}

function testText(entry: unknown): string | undefined {
  if (typeof entry === 'string') return entry.trim() || undefined;
  if (typeof entry !== 'object' || entry === null) return undefined;
  const item = entry as Record<string, unknown>;
  for (const field of [
    'input', 'content', 'user_input', 'agent_output', 'agent_message',
    'tool_response', 'tool_description', 'tool_name', 'tool_args',
  ]) {
    if (typeof item[field] === 'string' && item[field].trim().length > 0) return item[field].trim();
  }
  if (typeof item.tool_call === 'object' && item.tool_call !== null) return JSON.stringify(item.tool_call);
  const payload = Object.fromEntries(
    Object.entries(item).filter(([key]) => !['expected', 'description', 'notes'].includes(key)),
  );
  return Object.keys(payload).length > 0 ? JSON.stringify(payload) : undefined;
}

function collectSamples(rules: readonly ATRRule[]): { samples: Sample[]; skipped: number } {
  const byText = new Map<string, string[]>();
  let skipped = 0;
  for (const rule of rules) {
    const cases = rule.test_cases as unknown as Record<string, unknown[]> | undefined;
    for (const bucket of ['true_positives', 'true_negatives']) {
      for (const [index, entry] of (cases?.[bucket] ?? []).entries()) {
        const text = testText(entry);
        if (text === undefined) { skipped++; continue; }
        const source = `${rule.id}:${bucket}:${index}`;
        byText.set(text, [...(byText.get(text) ?? []), source]);
      }
    }
  }
  const samples = [...byText.entries()].map(([text, sources]) => ({
    id: sources[0]!,
    text,
    sources,
  }));
  return { samples, skipped };
}

function casesFor(samples: readonly Sample[]): EvaluationCase[] {
  const out: EvaluationCase[] = [];
  for (const sample of samples) {
    for (const shape of corpusShapes(sample.text)) {
      out.push({
        key: `${sample.id}\0${shape.name}`,
        sampleId: sample.id,
        text: sample.text,
        shape: shape.name,
        event: shape.event,
      });
    }
    out.push({
      key: `${sample.id}\0skill`,
      sampleId: sample.id,
      text: sample.text,
      shape: 'skill',
    });
  }
  return out;
}

function difference(baseline: readonly string[], actual: readonly string[]): Pick<Mismatch, 'missing' | 'unexpected'> {
  const baseSet = new Set(baseline);
  const actualSet = new Set(actual);
  return {
    missing: baseline.filter((id) => !actualSet.has(id)),
    unexpected: actual.filter((id) => !baseSet.has(id)),
  };
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(items: readonly T[], seed: number): T[] {
  const out = [...items];
  const random = mulberry32(seed);
  for (let index = out.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [out[index], out[other]] = [out[other]!, out[index]!];
  }
  return out;
}

function permutations<T>(items: readonly T[]): T[][] {
  if (items.length <= 1) return [[...items]];
  return items.flatMap((item, index) =>
    permutations([...items.slice(0, index), ...items.slice(index + 1)]).map((rest) => [item, ...rest]),
  );
}

function syntheticRule(id: string, pattern: string): ATRRule {
  return {
    title: id,
    id,
    status: 'stable',
    maturity: 'stable',
    description: 'Load-order experiment fixture',
    author: 'experiment',
    date: '2026/09/07',
    severity: 'medium',
    tags: { category: 'prompt-injection', confidence: 'high' },
    agent_source: { type: 'llm_io' },
    detection: {
      conditions: { selection: { field: 'user_input', patterns: [pattern], match_type: 'contains' } },
      condition: 'selection',
    },
    response: { actions: ['alert'] },
  };
}

async function verifyHarness(): Promise<void> {
  const fixtures = [
    syntheticRule('ATR-2026-99001', 'shared attack phrase'),
    syntheticRule('ATR-2026-99002', 'shared attack phrase'),
    syntheticRule('ATR-2026-99003', 'different phrase'),
  ];
  const event: AgentEvent = {
    type: 'llm_input',
    timestamp: '2026-09-07T00:00:00.000Z',
    content: 'shared attack phrase',
  };
  for (const order of permutations(fixtures)) {
    const result = ids((await engineFor(order)).evaluate(event));
    if (!sameIds(result, ['ATR-2026-99001', 'ATR-2026-99002'])) {
      throw new Error(`synthetic positive control failed: ${JSON.stringify(result)}`);
    }
  }

  const duplicateA = syntheticRule('ATR-2026-99990', 'first-only');
  const duplicateB = syntheticRule('ATR-2026-99990', 'second-only');
  const duplicateEvent: AgentEvent = { ...event, content: 'first-only' };
  const first = ids((await engineFor([duplicateA, duplicateB])).evaluate(duplicateEvent));
  const second = ids((await engineFor([duplicateB, duplicateA])).evaluate(duplicateEvent));
  if (sameIds(first, second)) throw new Error('duplicate-ID negative control did not detect order dependence');
}

function stableCaseOrder(item: EvaluationCase): string {
  return createHash('sha256').update(`${item.text}\0${item.shape}`).digest('hex');
}

function stratifiedCases(cases: readonly EvaluationCase[], baseline: ReadonlyMap<string, readonly string[]>): EvaluationCase[] {
  const buckets: EvaluationCase[][] = [[], [], []];
  for (const item of cases) {
    const count = baseline.get(item.key)?.length ?? 0;
    buckets[count === 0 ? 0 : count === 1 ? 1 : 2]!.push(item);
  }
  return buckets.flatMap((bucket) => bucket.sort((a, b) => stableCaseOrder(a).localeCompare(stableCaseOrder(b))).slice(0, 100));
}

async function baselineFor(rules: readonly ATRRule[], cases: readonly EvaluationCase[]): Promise<Map<string, string[]>> {
  const engine = await engineFor(rules);
  const baseline = new Map<string, string[]>();
  for (const [index, item] of cases.entries()) {
    baseline.set(item.key, evaluateCase(engine, item));
    if ((index + 1) % 1000 === 0) process.stderr.write(`baseline: ${index + 1}/${cases.length}\n`);
  }
  return baseline;
}

async function confirmedMismatch(
  baselineOrder: readonly ATRRule[],
  candidateOrder: readonly ATRRule[],
  item: EvaluationCase,
): Promise<{ baseline: string[]; actual: string[] } | undefined> {
  let last: { baseline: string[]; actual: string[] } | undefined;
  for (let attempt = 0; attempt < 3; attempt++) {
    const baseline = evaluateCase(await engineFor(baselineOrder), item);
    const actual = evaluateCase(await engineFor(candidateOrder), item);
    if (sameIds(baseline, actual)) return undefined;
    if (last !== undefined && (!sameIds(last.baseline, baseline) || !sameIds(last.actual, actual))) return undefined;
    last = { baseline, actual };
  }
  return last;
}

async function compareOrder(
  order: Order,
  cases: readonly EvaluationCase[],
  baseline: ReadonlyMap<string, readonly string[]>,
  baselineOrder: readonly ATRRule[],
): Promise<Mismatch | undefined> {
  process.stderr.write(`order ${order.name}: ${cases.length} cases\n`);
  const engine = await engineFor(order.rules);
  for (const [index, item] of cases.entries()) {
    const expected = baseline.get(item.key)!;
    const actual = evaluateCase(engine, item);
    if (!sameIds(expected, actual)) {
      const confirmed = await confirmedMismatch(baselineOrder, order.rules, item);
      if (confirmed !== undefined) {
        return {
          caseKey: item.key,
          sampleId: item.sampleId,
          shape: item.shape,
          order: order.name,
          baseline: confirmed.baseline,
          actual: confirmed.actual,
          ...difference(confirmed.baseline, confirmed.actual),
        };
      }
      process.stderr.write(`non-reproducible observation ignored: ${item.key}\n`);
    }
    if ((index + 1) % 1000 === 0) process.stderr.write(`  ${index + 1}/${cases.length}\n`);
  }
  return undefined;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const label = valueAfter(argv, '--label') ?? 'unnamed';
  const targetSha = valueAfter(argv, '--target-sha') ?? 'unknown';
  const rulesDir = resolve(valueAfter(argv, '--rules-dir') ?? 'rules');
  const output = valueAfter(argv, '--output');

  await verifyHarness();
  process.stderr.write('harness controls: PASS\n');

  const rules = loadRulesFromDirectory(rulesDir);
  const idsSeen = new Set<string>();
  const duplicates = [...new Set(rules.map((rule) => rule.id).filter((id) => idsSeen.has(id) || !idsSeen.add(id)))];
  if (duplicates.length > 0) throw new Error(`duplicate rule IDs: ${duplicates.join(', ')}`);

  const { samples, skipped } = collectSamples(rules);
  const allCases = casesFor(samples);
  process.stderr.write(`${label}: ${rules.length} rules, ${samples.length} unique samples, ${allCases.length} shape cases\n`);
  const baseline = await baselineFor(rules, allCases);
  const sampled = stratifiedCases(allCases, baseline);
  const bucketCounts = [0, 1, 2].map((bucket) => sampled.filter((item) => {
    const count = baseline.get(item.key)?.length ?? 0;
    return bucket === 0 ? count === 0 : bucket === 1 ? count === 1 : count >= 2;
  }).length);

  const structural: Order[] = [
    { name: 'reverse', rules: [...rules].reverse() },
    { name: 'id-ascending', rules: [...rules].sort((a, b) => a.id.localeCompare(b.id)) },
    { name: 'id-descending', rules: [...rules].sort((a, b) => b.id.localeCompare(a.id)) },
  ];
  for (const order of structural) {
    const mismatch = await compareOrder(order, allCases, baseline, rules);
    if (mismatch !== undefined) {
      const report = { status: 'counterexample', label, targetSha, ruleCount: rules.length, sampleCount: samples.length, mismatch };
      if (output !== undefined) writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
      process.exit(2);
    }
  }

  for (let seed = 1; seed <= 20; seed++) {
    const mismatch = await compareOrder({ name: `shuffle-seed-${seed}`, rules: shuffled(rules, seed) }, sampled, baseline, rules);
    if (mismatch !== undefined) {
      const report = { status: 'counterexample', label, targetSha, ruleCount: rules.length, sampleCount: samples.length, mismatch };
      if (output !== undefined) writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
      process.exit(2);
    }
  }

  const report = {
    status: 'no-counterexample',
    label,
    targetSha,
    ruleCount: rules.length,
    uniqueSampleCount: samples.length,
    skippedFixtureCount: skipped,
    shapeCaseCount: allCases.length,
    structuralOrders: ['original', ...structural.map((order) => order.name)],
    randomShuffleCount: 20,
    stratifiedShapeCases: { zeroMatch: bucketCounts[0], singleMatch: bucketCounts[1], multiMatch: bucketCounts[2] },
  };
  if (output !== undefined) writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`order experiment failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exit(1);
});
