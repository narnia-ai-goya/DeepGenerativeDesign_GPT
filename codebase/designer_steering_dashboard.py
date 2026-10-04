#!/usr/bin/env python3
"""Local web UI for auditable designer interventions in Semantic BO-QD."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from designer_steering import apply_action, load_state, save_state


PAGE = r'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Designer-Steered Semantic BO-QD</title>
<style>
:root{--navy:#133757;--teal:#087b6b;--orange:#e26d2f;--line:#cbd8e2;--bg:#edf3f6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#172b3a;font:15px system-ui,sans-serif}
header{background:var(--navy);color:white;padding:22px 30px}h1{margin:0 0 4px;font-size:26px}header p{margin:0;color:#cce0ed}
main{max-width:1500px;margin:22px auto;padding:0 18px}.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}
section{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:18px;box-shadow:0 4px 18px #17324710}
h2{margin:0 0 13px;color:var(--navy);font-size:18px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.step{padding:13px 9px;border:1px solid #bcd2df;border-radius:9px;text-align:center;background:#f7fbfd}.step b{display:block;color:#126e8d}.hard{border-color:#efb49a;background:#fff6f1}.hard b{color:#b74018}
label{display:block;font-size:12px;color:#526b7d;margin-top:10px}input,select,button{font:inherit}input,select{width:100%;padding:9px;border:1px solid #b7c7d2;border-radius:6px;margin-top:4px}.row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:12px}.action{border:1px solid #d4dfe6;padding:12px;border-radius:9px}button{border:0;border-radius:7px;background:var(--teal);color:white;padding:9px 13px;margin-top:10px;cursor:pointer}.danger{background:#a8462b}.badge{display:inline-block;padding:4px 8px;border-radius:20px;background:#e8f4f1;margin:3px;font-size:12px}.gate{color:#1d6d52}.gate:before{content:'✓ ';font-weight:bold}.log{max-height:290px;overflow:auto;font:12px ui-monospace,monospace;background:#f6f8fa;padding:10px;border-radius:7px}.muted{color:#647b8b}.status{position:fixed;right:20px;bottom:20px;background:#142f43;color:white;padding:10px 15px;border-radius:8px;display:none}@media(max-width:900px){.grid,.actions{grid-template-columns:1fr}.flow{grid-template-columns:1fr 1fr}}
</style>
<header><h1>Designer-Steered Semantic BO-QD</h1><p>Soft preference steering with immutable automatic physics gates</p></header>
<main><section><div class="flow"><div class="step"><b>1 · Designer intent</b>anchors · mass range</div><div class="step"><b>2 · BO-QD</b>target · lock · reject</div><div class="step hard"><b>3 · Hard gates</b>BC · geometry · FEA</div><div class="step"><b>4 · Archive</b>semantic × mass</div></div></section>
<div class="grid"><div>
<section><h2>Designer interventions</h2><div class="actions">
<div class="action"><b>Target archive cell</b><div class="row"><label>Semantic niche<input id="niche" value="arch"></label><label>Mass bin<input id="massbin" value="medium"></label><label>Weight<input id="cellweight" type="number" value="2" step="0.1"></label></div><button onclick="targetCell()">Set target</button></div>
<div class="action"><b>Preferred mass range</b><div class="row"><label>Low<input id="masslow" type="number" min="0" max="1" step=".01"></label><label>High<input id="masshigh" type="number" min="0" max="1" step=".01"></label></div><button onclick="massRange()">Update range</button></div>
<div class="action"><b>Semantic anchor</b><div class="row"><label>Anchor<input id="anchor" value="continuous_load_path"></label><label>Weight<input id="anchorweight" type="number" value="0.8" step=".1"></label></div><button onclick="anchorWeight()">Set anchor</button></div>
<div class="action"><b>Candidate decision</b><label>Candidate ID<input id="candidate" placeholder="candidate_012"></label><button onclick="candidateAction('lock_candidate')">Lock</button> <button class="danger" onclick="candidateAction('reject_candidate')">Reject</button> <button onclick="explore()">Explore locally</button></div>
</div></section>
<section><h2>Archive steering state</h2><div id="summary"></div></section>
</div><div>
<section class="hard"><h2>Automatic hard gates · read only</h2><p class="muted">Designer preferences cannot bypass these checks.</p><div id="gates"></div></section>
<section><h2>Interaction trace</h2><div id="log" class="log"></div></section>
</div></div></main><div id="status" class="status"></div>
<script>
let state={};
async function refresh(){state=await (await fetch('/api/state')).json();render()}
function render(){masslow.value=state.target_mass_range[0];masshigh.value=state.target_mass_range[1];gates.innerHTML=state.hard_gates.checks.map(x=>`<p class="gate">${x.replaceAll('_',' ')}</p>`).join('');summary.innerHTML=`<p><b>Revision:</b> ${state.revision} · <b>Axes:</b> ${state.archive_axes.y} × ${state.archive_axes.x}</p><p><b>Target cells</b><br>${Object.entries(state.target_cells).map(([k,v])=>`<span class="badge">${k} ×${v}</span>`).join('')||'<span class="muted">none</span>'}</p><p><b>Locked:</b> ${state.locked_candidates.join(', ')||'none'}<br><b>Rejected:</b> ${state.rejected_candidates.join(', ')||'none'}</p>`;log.textContent=state.interaction_log.slice().reverse().map(x=>JSON.stringify(x)).join('\n\n')||'No intervention yet.'}
async function act(action){let r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(action)});let data=await r.json();if(!r.ok){notify(data.error);return}state=data;render();notify('Saved revision '+state.revision)}
function targetCell(){act({type:'set_target_cell',semantic_niche:niche.value,mass_bin:massbin.value,weight:+cellweight.value})}
function massRange(){act({type:'set_mass_range',low:+masslow.value,high:+masshigh.value})}
function anchorWeight(){act({type:'set_anchor_weight',anchor:anchor.value,weight:+anchorweight.value})}
function candidateAction(type){act({type,candidate_id:candidate.value})}
function explore(){act({type:'request_local_exploration',candidate_id:candidate.value,radius:.15,weight:1.5})}
function notify(x){status.textContent=x;status.style.display='block';setTimeout(()=>status.style.display='none',2600)}refresh();
</script></html>'''


def handler_for(state_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/": self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif route == "/api/state": self.send(200, json.dumps(load_state(state_path), ensure_ascii=False).encode(), "application/json")
            else: self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/action": self.send(404, b"not found", "text/plain"); return
            try:
                length = int(self.headers.get("Content-Length", 0))
                action = json.loads(self.rfile.read(length))
                state = apply_action(load_state(state_path), action)
                save_state(state_path, state)
                self.send(200, json.dumps(state, ensure_ascii=False).encode(), "application/json")
            except Exception as exc:
                self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

        def log_message(self, fmt, *args):
            print(f"designer-ui: {fmt % args}")
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    load_state(args.state)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(args.state.resolve()))
    print(f"http://{args.host}:{args.port}/")
    print(args.state.resolve())
    server.serve_forever()


if __name__ == "__main__":
    main()
