#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a lightweight labeling HTML.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint with class list.")
    parser.add_argument("--out", default="/home/jaimelop/spacenet/reports/active_labeler.html")
    parser.add_argument("--csv-path", default="/home/jaimelop/spacenet/logs/active_learning_batch_urls.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt.get("classes", None)
    if not classes:
        raise SystemExit("Checkpoint missing classes.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Active Learning Labeler</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f6f7fb; }}
    .panel {{ background: #fff; border: 1px solid #e4e7ef; border-radius: 10px; padding: 16px; margin-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #eef1f6; padding: 6px; text-align: left; }}
    select {{ padding: 4px; }}
    button {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #d0d5dd; background: #fff; cursor: pointer; }}
    .thumb {{ width: 120px; height: 90px; object-fit: cover; border: 1px solid #e5e7ef; border-radius: 6px; }}
  </style>
</head>
<body>
    <div class="panel">
    <h2 style="margin-top:0">Active Learning Labeler</h2>
    <p>Load the CSV from <code>logs/active_learning_batch.csv</code>, assign labels, then export.</p>
    <div style="margin-bottom:8px;">
      <a href="ood_gallery.html" style="font-size:12px;text-decoration:none;color:#2563eb;">Open OOD Gallery</a>
    </div>
    <input type="file" id="fileInput" accept=".csv" />
    <button id="exportBtn">Export CSV</button>
  </div>
  <div class="panel">
    <table id="table">
      <thead>
        <tr><th>preview</th><th>path</th><th>pred_class</th><th>confidence</th><th>entropy</th><th>label</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
  <script>
    const classes = {json.dumps(classes)};
    const fileInput = document.getElementById('fileInput');
    const tbody = document.querySelector('#table tbody');
    let rows = [];

    function render() {{
      tbody.innerHTML = '';
      rows.forEach((row, idx) => {{
        const tr = document.createElement('tr');
        const imgTd = document.createElement('td');
        const img = document.createElement('img');
        img.src = row.path;
        img.className = 'thumb';
        imgTd.appendChild(img);
        tr.appendChild(imgTd);
        ['path','pred_class','confidence','entropy'].forEach(key => {{
          const td = document.createElement('td');
          td.textContent = row[key];
          tr.appendChild(td);
        }});
        const td = document.createElement('td');
        const sel = document.createElement('select');
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '';
        sel.appendChild(empty);
        classes.forEach(c => {{
          const opt = document.createElement('option');
          opt.value = c;
          opt.textContent = c;
          if (row.label === c) opt.selected = true;
          sel.appendChild(opt);
        }});
        sel.addEventListener('change', () => {{
          rows[idx].label = sel.value;
        }});
        td.appendChild(sel);
        tr.appendChild(td);
        tbody.appendChild(tr);
      }});
    }}

    fileInput.addEventListener('change', (e) => {{
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {{
        const lines = reader.result.trim().split(/\\r?\\n/);
        const header = lines.shift().split(',');
        rows = lines.map(line => {{
          const parts = line.split(',');
          const obj = {{}};
          header.forEach((h, i) => obj[h] = parts[i] || '');
          return obj;
        }});
        render();
      }};
      reader.readAsText(file);
    }});

    document.getElementById('exportBtn').addEventListener('click', () => {{
      const header = ['path','pred_class','confidence','entropy','label'];
      const out = [header.join(',')].concat(rows.map(r => header.map(k => r[k] || '').join(','))).join('\\n');
      const blob = new Blob([out], {{ type: 'text/csv' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'active_learning_labeled.csv';
      a.click();
      URL.revokeObjectURL(url);
    }});

    // Auto-load default CSV if present
    fetch('{args.csv_path}')
      .then(r => r.ok ? r.text() : null)
      .then(text => {{
        if (!text) return;
        const lines = text.trim().split(/\\r?\\n/);
        const header = lines.shift().split(',');
        rows = lines.map(line => {{
          const parts = line.split(',');
          const obj = {{}};
          header.forEach((h, i) => obj[h] = parts[i] || '');
          return obj;
        }});
        render();
      }})
      .catch(() => {{}});
  </script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
