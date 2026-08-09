#!/usr/bin/env python3
"""Claude Code token usage analytics.

Usage:
    python3 scripts/token_analytics.py              # all projects
    python3 scripts/token_analytics.py --project nas # specific project
    python3 scripts/token_analytics.py --session ID  # deep dive one session
"""
import json, os, glob, argparse
from collections import defaultdict
from datetime import datetime

PRICES = {
    'opus':   {'input': 15.0/1e6, 'output': 75.0/1e6, 'cache_write': 18.75/1e6, 'cache_read': 1.50/1e6},
    'sonnet': {'input': 3.0/1e6,  'output': 15.0/1e6, 'cache_write': 3.75/1e6,  'cache_read': 0.30/1e6},
    'haiku':  {'input': 0.80/1e6, 'output': 4.0/1e6,  'cache_write': 1.0/1e6,   'cache_read': 0.08/1e6},
}

def detect_model(model_str):
    m = model_str.lower()
    if 'opus' in m: return 'opus'
    if 'haiku' in m: return 'haiku'
    return 'sonnet'

def calc_cost(usage, model_str):
    mn = detect_model(model_str)
    p = PRICES[mn]
    inp = usage.get('input_tokens', 0)
    cw = usage.get('cache_creation_input_tokens', 0)
    cr = usage.get('cache_read_input_tokens', 0)
    out = usage.get('output_tokens', 0)
    cost = inp*p['input'] + cw*p['cache_write'] + cr*p['cache_read'] + out*p['output']
    return cost, mn, inp, cw, cr, out

def analyze_session(filepath):
    totals = defaultdict(float)
    model_stats = defaultdict(lambda: defaultdict(float))
    api_calls = 0
    user_msgs = 0
    first_ts = last_ts = None
    context_growth = []

    with open(filepath) as fh:
        for line in fh:
            try:
                msg = json.loads(line)
            except:
                continue
            ts = msg.get('timestamp')
            if ts:
                if not first_ts: first_ts = ts
                last_ts = ts
            if msg.get('type') == 'user' and msg.get('message', {}).get('role') == 'user':
                user_msgs += 1
            if msg.get('type') != 'assistant':
                continue
            m = msg.get('message', {})
            usage = m.get('usage', {})
            if not usage:
                continue
            model = m.get('model', 'unknown')
            cost, mn, inp, cw, cr, out = calc_cost(usage, model)
            api_calls += 1
            totals['cost'] += cost
            totals['input'] += inp
            totals['cache_write'] += cw
            totals['cache_read'] += cr
            totals['output'] += out
            totals['total_input'] += inp + cw + cr
            model_stats[mn]['cost'] += cost
            model_stats[mn]['calls'] += 1
            model_stats[mn]['output'] += out
            context_growth.append((api_calls, inp+cw+cr, mn, cost, out))

    duration_h = 0
    if first_ts and last_ts:
        try:
            t1 = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
            t2 = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            duration_h = (t2 - t1).total_seconds() / 3600
        except:
            pass

    return {
        'totals': dict(totals), 'model_stats': dict(model_stats),
        'api_calls': api_calls, 'user_msgs': user_msgs,
        'duration_h': duration_h, 'context_growth': context_growth,
    }

def print_session_detail(filepath):
    r = analyze_session(filepath)
    sid = os.path.basename(filepath)[:12]
    t = r['totals']
    print(f"\n{'SESSION DETAIL: ' + sid:>50}")
    print("=" * 80)
    print(f"  Duration:        {r['duration_h']:.1f}h")
    print(f"  User messages:   {r['user_msgs']}")
    print(f"  API calls:       {r['api_calls']}")
    print(f"  Total input:     {int(t.get('total_input',0)):,} tokens")
    print(f"  Total output:    {int(t.get('output',0)):,} tokens")
    print(f"  Cache hit rate:  {t.get('cache_read',0)/max(t.get('total_input',1),1)*100:.1f}%")
    print(f"  Est. cost:       ${t.get('cost',0):.2f}")

    print(f"\n  By Model:")
    for mn, stats in sorted(r['model_stats'].items(), key=lambda x: x[1]['cost'], reverse=True):
        pct = stats['cost'] / max(t.get('cost', 1), 0.01) * 100
        avg_out = int(stats['output'] / max(stats['calls'], 1))
        print(f"    {mn:>8}: {int(stats['calls']):>4} calls, ${stats['cost']:>8.2f} ({pct:.0f}%), avg output: {avg_out:,}/call")

    cg = r['context_growth']
    if cg:
        print(f"\n  Context Growth:")
        indices = [0, len(cg)//4, len(cg)//2, 3*len(cg)//4, len(cg)-1]
        for i in indices:
            cn, ti, mn, c, out = cg[i]
            print(f"    Call #{cn:>4}: {ti:>10,} input, {out:>6,} output ({mn}) ${c:.2f}")

        print(f"\n  Top 5 Most Expensive Calls:")
        for cn, ti, mn, c, out in sorted(cg, key=lambda x: x[3], reverse=True)[:5]:
            print(f"    Call #{cn:>4}: ${c:.2f} — {ti:,} in, {out:,} out ({mn})")

def main():
    parser = argparse.ArgumentParser(description='Claude Code Token Analytics')
    parser.add_argument('--project', '-p', help='Filter by project name substring')
    parser.add_argument('--session', '-s', help='Deep dive a specific session ID prefix')
    args = parser.parse_args()

    projects_dir = os.path.expanduser('~/.claude/projects/')
    proj_dirs = sorted(glob.glob(os.path.join(projects_dir, '*/')))

    if args.project:
        proj_dirs = [d for d in proj_dirs if args.project.lower() in d.lower()]

    grand_cost = 0
    grand_calls = 0

    print("=" * 80)
    print("CLAUDE CODE TOKEN USAGE ANALYTICS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    for proj_dir in proj_dirs:
        proj_name = os.path.basename(proj_dir.rstrip('/'))
        jsonl_files = sorted(glob.glob(os.path.join(proj_dir, '*.jsonl')), key=os.path.getmtime)

        if not jsonl_files:
            continue

        proj_cost = 0
        proj_calls = 0
        sessions = []

        for f in jsonl_files:
            if args.session:
                sid = os.path.basename(f).replace('.jsonl', '')
                if sid.startswith(args.session):
                    print_session_detail(f)
                    return

            r = analyze_session(f)
            if r['api_calls'] == 0:
                continue
            t = r['totals']
            proj_cost += t.get('cost', 0)
            proj_calls += r['api_calls']
            sessions.append((f, r))

        grand_cost += proj_cost
        grand_calls += proj_calls

        print(f"\nProject: {proj_name}")
        print(f"  Sessions: {len(sessions)}  |  API calls: {proj_calls:,}  |  Est. cost: ${proj_cost:.2f}")
        print(f"  {'Session':<14} {'Dur':>5} {'Msgs':>5} {'Calls':>6} {'Cost':>9} {'Models'}")
        for f, r in sorted(sessions, key=lambda x: x[1]['totals'].get('cost', 0), reverse=True):
            sid = os.path.basename(f)[:12]
            t = r['totals']
            models = ', '.join(f"{k}:{int(v['calls'])}" for k, v in r['model_stats'].items())
            print(f"  {sid:<14} {r['duration_h']:>4.1f}h {r['user_msgs']:>5} {r['api_calls']:>6} ${t.get('cost',0):>8.2f} {models}")

    print(f"\n{'GRAND TOTAL':>40}: ${grand_cost:.2f} across {grand_calls:,} API calls")
    sonnet_estimate = grand_cost * 0.41  # rough ratio from actual data
    print(f"{'Estimated on all-Sonnet':>40}: ~${sonnet_estimate:.2f} (save ~{(1-0.41)*100:.0f}%)")

if __name__ == '__main__':
    main()
